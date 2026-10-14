import * as ort from "onnxruntime-web";
import { decodeYuNet } from "../yunetDecode";
import type { DetectedFace, DetectorTimings, FaceDetector } from "./types";

const CONFIDENCE_THRESHOLD = 0.6;
const MODEL_PATH = "/models/yunet.onnx";

export class YuNetDetector implements FaceDetector {
  readonly id = "yunet";
  readonly name = "YuNet";
  readonly inputSize = { width: 640, height: 640 } as const;

  private session: ort.InferenceSession | null = null;
  private loadPromise: Promise<void> | null = null;
  private activeInference: Promise<{
    faces: DetectedFace[];
    timings: DetectorTimings;
  }> | null = null;
  private disposePromise: Promise<void> | null = null;
  private disposed = false;
  private sourceCanvas: HTMLCanvasElement | null = null;
  private modelCanvas: HTMLCanvasElement | null = null;
  private inputData: Float32Array | null = null;

  load(): Promise<void> {
    if (this.disposed) return Promise.reject(new Error("Detector is disposed"));
    if (this.session) return Promise.resolve();
    if (this.loadPromise) return this.loadPromise;

    const loading = ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    }).then(async (session) => {
      if (this.disposed) {
        await session.release();
        throw new Error("Detector was disposed during load");
      }
      this.session = session;
    });

    this.loadPromise = loading;
    void loading.then(
      () => {
        if (this.loadPromise === loading) this.loadPromise = null;
      },
      () => {
        if (this.loadPromise === loading) this.loadPromise = null;
      }
    );
    return loading;
  }

  async detect(image: ImageData) {
    if (this.disposed) throw new Error("Detector is disposed");
    if (!this.session) throw new Error("Detector must be loaded before detect()");
    if (this.activeInference) throw new Error("Inference is already in progress");

    const inference = this.infer(this.session, image);
    this.activeInference = inference;
    void inference.then(
      () => {
        if (this.activeInference === inference) this.activeInference = null;
      },
      () => {
        if (this.activeInference === inference) this.activeInference = null;
      }
    );
    return inference;
  }

  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;

    const disposal = (async () => {
      await this.loadPromise?.catch(() => undefined);
      await this.activeInference?.catch(() => undefined);
      const session = this.session;
      this.session = null;
      await session?.release();
    })();

    this.disposePromise = disposal;
    return disposal;
  }

  private async infer(session: ort.InferenceSession, image: ImageData) {
    if (image.width <= 0 || image.height <= 0) {
      throw new Error("Image dimensions must be positive");
    }

    const { width: modelSize, height: modelHeight } = this.inputSize;
    const sourceCanvas = this.sourceCanvas ??= document.createElement("canvas");
    const modelCanvas = this.modelCanvas ??= document.createElement("canvas");
    if (sourceCanvas.width !== image.width) sourceCanvas.width = image.width;
    if (sourceCanvas.height !== image.height) sourceCanvas.height = image.height;
    if (modelCanvas.width !== modelSize) modelCanvas.width = modelSize;
    if (modelCanvas.height !== modelHeight) modelCanvas.height = modelHeight;

    const sourceContext = sourceCanvas.getContext("2d", {
      willReadFrequently: true,
    });
    const modelContext = modelCanvas.getContext("2d", {
      willReadFrequently: true,
    });
    if (!sourceContext || !modelContext) {
      throw new Error("Could not initialize detector canvas");
    }

    const startedAt = performance.now();
    sourceContext.putImageData(image, 0, 0);

    const scale = Math.min(modelSize / image.width, modelHeight / image.height);
    const width = image.width * scale;
    const height = image.height * scale;
    const offsetX = (modelSize - width) / 2;
    const offsetY = (modelHeight - height) / 2;

    modelContext.fillStyle = "black";
    modelContext.fillRect(0, 0, modelSize, modelHeight);
    modelContext.drawImage(sourceCanvas, offsetX, offsetY, width, height);

    const rgba = modelContext.getImageData(0, 0, modelSize, modelHeight).data;
    const pixels = modelSize * modelHeight;
    const inputData = this.inputData ??= new Float32Array(3 * pixels);
    for (let i = 0; i < pixels; i++) {
      const pixel = i * 4;
      inputData[i] = rgba[pixel + 2];
      inputData[pixels + i] = rgba[pixel + 1];
      inputData[2 * pixels + i] = rgba[pixel];
    }

    const tensor = new ort.Tensor("float32", inputData, [1, 3, modelHeight, modelSize]);
    const preprocessingMs = performance.now() - startedAt;
    let outputs: Awaited<ReturnType<ort.InferenceSession["run"]>> | null = null;

    try {
      const inferenceStartedAt = performance.now();
      outputs = await session.run({ input: tensor });
      const inferenceMs = performance.now() - inferenceStartedAt;

      const decodingStartedAt = performance.now();
      const faces = decodeYuNet(
        outputs,
        {
          scale,
          offsetX,
          offsetY,
          sourceWidth: image.width,
          sourceHeight: image.height,
        },
        CONFIDENCE_THRESHOLD
      );
      const decodingMs = performance.now() - decodingStartedAt;

      return {
        faces,
        timings: { preprocessingMs, inferenceMs, decodingMs },
      };
    } finally {
      if (outputs) Object.values(outputs).forEach((output) => output.dispose());
      tensor.dispose();
    }
  }
}
