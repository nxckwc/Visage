
import * as ort from "onnxruntime-web";
import type {
  FaceDetector,
  DetectedFace,
  DetectorTimings,
} from "./types";

const SIZE = 640;
const THRESHOLD = 0.6;
const NMS_THRESHOLD = 0.4;

type DetectionResult = {
  faces: DetectedFace[];
  timings: DetectorTimings;
};

function calculateIoU(a: DetectedFace, b: DetectedFace) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);

  const intersection =
    Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

  const union =
    a.width * a.height +
    b.width * b.height -
    intersection;

  return union > 0 ? intersection / union : 0;
}

function nms(faces: DetectedFace[]): DetectedFace[] {
  const sorted = [...faces].sort(
    (a, b) => b.confidence - a.confidence
  );

  const selected: DetectedFace[] = [];

  for (const face of sorted) {
    if (
      selected.every(
        (existing) =>
          calculateIoU(existing, face) < NMS_THRESHOLD
      )
    ) {
      selected.push(face);
    }

    if (selected.length >= 100) break;
  }

  return selected;
}

export class SCRFDDetector implements FaceDetector {
  readonly id = "scrfd";
  readonly name = "SCRFD-500M";
  readonly inputSize = { width: SIZE, height: SIZE };

  private session: ort.InferenceSession | null = null;
  private loadPromise: Promise<void> | null = null;
  private activeDetection: Promise<DetectionResult> | null = null;
  private disposePromise: Promise<void> | null = null;
  private disposed = false;
  private sourceCanvas: HTMLCanvasElement | null = null;
  private modelCanvas: HTMLCanvasElement | null = null;
  private inputData: Float32Array | null = null;

  async load(): Promise<void> {
    if (this.disposed) {
      throw new Error("SCRFD detector has been disposed");
    }

    if (this.session) return;
    if (this.loadPromise) return this.loadPromise;

    this.loadPromise = (async () => {
      const session = await ort.InferenceSession.create(
        "/models/scrfd-500m.onnx",
        { executionProviders: ["wasm"] }
      );

      if (this.disposed) {
        await session.release();
        throw new Error("SCRFD detector was disposed during load");
      }

      this.session = session;
    })();

    try {
      await this.loadPromise;
    } finally {
      this.loadPromise = null;
    }
  }

  detect(image: ImageData): Promise<DetectionResult> {
    if (this.disposed) {
      return Promise.reject(
        new Error("SCRFD detector has been disposed")
      );
    }

    if (!this.session) {
      return Promise.reject(
        new Error("SCRFD is not loaded")
      );
    }

    if (this.activeDetection) {
      return Promise.reject(
        new Error("SCRFD is already processing a frame")
      );
    }

    const work = this.detectInternal(image);
    this.activeDetection = work;

    void work.finally(() => {
      if (this.activeDetection === work) {
        this.activeDetection = null;
      }
    }).catch(() => {});

    return work;
  }

  private async detectInternal(
    image: ImageData
  ): Promise<DetectionResult> {
    if (image.width <= 0 || image.height <= 0) {
      throw new Error("Image dimensions must be positive");
    }
    const session = this.session;
    if (!session) throw new Error("SCRFD is not loaded");

    const preprocessStart = performance.now();

    const modelCanvas = this.modelCanvas ??= document.createElement("canvas");
    if (modelCanvas.width !== SIZE) modelCanvas.width = SIZE;
    if (modelCanvas.height !== SIZE) modelCanvas.height = SIZE;
    const modelContext = modelCanvas.getContext("2d", {
      willReadFrequently: true,
    });

    if (!modelContext) throw new Error("Could not initialize SCRFD model canvas");

    const scale = Math.min(
      SIZE / image.width,
      SIZE / image.height
    );

    const width = image.width * scale;
    const height = image.height * scale;

    const offsetX = (SIZE - width) / 2;
    const offsetY = (SIZE - height) / 2;

    const sourceCanvas = this.sourceCanvas ??= document.createElement("canvas");
    if (sourceCanvas.width !== image.width) sourceCanvas.width = image.width;
    if (sourceCanvas.height !== image.height) sourceCanvas.height = image.height;
    const sourceContext = sourceCanvas.getContext("2d", {
      willReadFrequently: true,
    });
    if (!sourceContext) throw new Error("Could not initialize SCRFD source canvas");

    sourceContext.putImageData(image, 0, 0);
    modelContext.fillStyle = "black";
    modelContext.fillRect(0, 0, SIZE, SIZE);
    modelContext.drawImage(sourceCanvas, offsetX, offsetY, width, height);

    const rgba = modelContext.getImageData(0, 0, SIZE, SIZE).data;
    const pixels = SIZE * SIZE;
    const data = this.inputData ??= new Float32Array(3 * pixels);

    // SCRFD expects RGB, CHW, normalized using
    // (pixel - 127.5) / 128.
    for (let i = 0; i < pixels; i++) {
      const p = i * 4;

      data[i] = (rgba[p] - 127.5) / 128;
      data[pixels + i] = (rgba[p + 1] - 127.5) / 128;
      data[2 * pixels + i] = (rgba[p + 2] - 127.5) / 128;
    }

    const tensor = new ort.Tensor(
      "float32",
      data,
      [1, 3, SIZE, SIZE]
    );

    const preprocessingMs =
      performance.now() - preprocessStart;

    let outputs: Awaited<
      ReturnType<ort.InferenceSession["run"]>
    > | null = null;

    try {
      const inferenceStart = performance.now();

      outputs = await session.run({ "input.1": tensor });

      const inferenceMs =
        performance.now() - inferenceStart;

      const decodeStart = performance.now();

      const faces: DetectedFace[] = [];

      const toSourceX = (x: number) =>
        (x - offsetX) / scale;

      const toSourceY = (y: number) =>
        (y - offsetY) / scale;

      const clamp = (value: number, max: number) =>
        Math.max(0, Math.min(max, value));

      const strides = [8, 16, 32];

      const scoreNames = ["443", "468", "493"];
      const boxNames = ["446", "471", "496"];
      const landmarkNames = ["449", "474", "499"];

      for (let level = 0; level < 3; level++) {
        const stride = strides[level];
        const gridWidth = SIZE / stride;

        const scoreOutput = outputs[scoreNames[level]];
        const boxOutput = outputs[boxNames[level]];
        const landmarkOutput = outputs[landmarkNames[level]];
        if (!scoreOutput || !boxOutput || !landmarkOutput) {
          throw new Error(`SCRFD output tensors are missing for stride ${stride}`);
        }
        if (
          !(scoreOutput.data instanceof Float32Array) ||
          !(boxOutput.data instanceof Float32Array) ||
          !(landmarkOutput.data instanceof Float32Array)
        ) {
          throw new Error(`SCRFD output tensors have an unexpected type for stride ${stride}`);
        }

        const scores = scoreOutput.data;
        const boxes = boxOutput.data;
        const keypoints = landmarkOutput.data;
        if (boxes.length < scores.length * 4 || keypoints.length < scores.length * 10) {
          throw new Error(`SCRFD output tensor shapes do not match for stride ${stride}`);
        }

        for (let i = 0; i < scores.length; i++) {
          const confidence = scores[i];

          if (!Number.isFinite(confidence) || confidence < THRESHOLD) continue;

          // Two anchors per feature-map position.
          const gridIndex = Math.floor(i / 2);
          const col = gridIndex % gridWidth;
          const row = Math.floor(gridIndex / gridWidth);

          const anchorX = col * stride;
          const anchorY = row * stride;

          const b = i * 4;

          const x1 = toSourceX(
            anchorX - boxes[b] * stride
          );

          const y1 = toSourceY(
            anchorY - boxes[b + 1] * stride
          );

          const x2 = toSourceX(
            anchorX + boxes[b + 2] * stride
          );

          const y2 = toSourceY(
            anchorY + boxes[b + 3] * stride
          );
          if (![x1, y1, x2, y2].every(Number.isFinite)) continue;

          const x = clamp(x1, image.width);
          const y = clamp(y1, image.height);

          const right = clamp(x2, image.width);
          const bottom = clamp(y2, image.height);

          if (right <= x || bottom <= y) continue;

          const landmarks: DetectedFace["landmarks"] = [];
          for (let j = 0; j < 5; j++) {
            const k = i * 10 + j * 2;
            const modelX = anchorX + keypoints[k] * stride;
            const modelY = anchorY + keypoints[k + 1] * stride;
            if (!Number.isFinite(modelX) || !Number.isFinite(modelY)) {
              landmarks.length = 0;
              break;
            }
            landmarks.push({
              x: clamp(toSourceX(modelX), image.width),
              y: clamp(toSourceY(modelY), image.height),
            });
          }
          if (landmarks.length !== 5) continue;

          faces.push({
            x,
            y,
            width: right - x,
            height: bottom - y,
            confidence,
            landmarks,
          });
        }
      }

      const selectedFaces = nms(faces);

      const decodingMs =
        performance.now() - decodeStart;

      return {
        faces: selectedFaces,
        timings: {
          preprocessingMs,
          inferenceMs,
          decodingMs,
        },
      };
    } finally {
      if (outputs) {
        Object.values(outputs).forEach(
          (output) => output.dispose()
        );
      }

      tensor.dispose();
    }
  }

  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;

    this.disposed = true;

    this.disposePromise = (async () => {
      if (this.loadPromise) {
        await this.loadPromise.catch(() => {});
      }

      if (this.activeDetection) {
        await this.activeDetection.catch(() => {});
      }

      if (this.session) {
        await this.session.release();
        this.session = null;
      }
      this.sourceCanvas = null;
      this.modelCanvas = null;
      this.inputData = null;
    })();

    return this.disposePromise;
  }
}
