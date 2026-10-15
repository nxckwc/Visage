import * as ort from "onnxruntime-web";
import type {
  DetectedFace,
  DetectorTimings,
  FaceDetector,
} from "./types";

type ModelLayout = "nhwc" | "nchw";

type DetectorConfig = {
  id: string;
  name: string;
  modelPath: string;
  inputSize: number;
  layout: ModelLayout;
  confidenceThreshold?: number;
  externalDataPath?: string;
  landmarkModelPath?: string;
  landmarkDataPath?: string;
};

type FrameTransform = {
  scale: number;
  offsetX: number;
  offsetY: number;
};

type CandidateFace = DetectedFace & {
  landmarks: { x: number; y: number }[];
};

const FACE_THRESHOLD = 0.5;
const NMS_THRESHOLD = 0.3;
const LANDMARK_THRESHOLD = 0.5;
const MAX_FACES = 10;
const LANDMARK_INPUT_SIZE = 192;

const ANCHORS = createAnchors();

class BlazeFaceOnnxDetector implements FaceDetector {
  private readonly config: DetectorConfig;
  readonly id: string;
  readonly name: string;
  readonly inputSize: { width: number; height: number };

  private detectorSession: ort.InferenceSession | null = null;
  private landmarkSession: ort.InferenceSession | null = null;
  private loadPromise: Promise<void> | null = null;
  private activeInference: Promise<{
    faces: DetectedFace[];
    timings: DetectorTimings;
  }> | null = null;
  private disposePromise: Promise<void> | null = null;
  private disposed = false;
  private sourceCanvas: HTMLCanvasElement | null = null;
  private modelCanvas: HTMLCanvasElement | null = null;
  private detectorInputData: Float32Array | null = null;
  private landmarkInputData: Float32Array | null = null;

  constructor(config: DetectorConfig) {
    this.config = config;
    this.id = config.id;
    this.name = config.name;
    this.inputSize = {
      width: config.inputSize,
      height: config.inputSize,
    };
  }

  load(): Promise<void> {
    if (this.disposed) {
      return Promise.reject(new Error(`${this.name} has been disposed`));
    }
    if (this.detectorSession && (!this.config.landmarkModelPath || this.landmarkSession)) {
      return Promise.resolve();
    }
    if (this.loadPromise) return this.loadPromise;

    const loading = this.loadSessions();
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

  detect(image: ImageData): Promise<{
    faces: DetectedFace[];
    timings: DetectorTimings;
  }> {
    if (this.disposed) {
      return Promise.reject(new Error(`${this.name} has been disposed`));
    }
    if (!this.detectorSession) {
      return Promise.reject(new Error(`${this.name} must be loaded before detect()`));
    }
    if (this.activeInference) {
      return Promise.reject(new Error(`${this.name} is already processing a frame`));
    }

    const work = this.detectFrame(image);
    this.activeInference = work;
    void work.then(
      () => {
        if (this.activeInference === work) this.activeInference = null;
      },
      () => {
        if (this.activeInference === work) this.activeInference = null;
      }
    );
    return work;
  }

  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;

    const disposal = (async () => {
      await this.loadPromise?.catch(() => undefined);
      await this.activeInference?.catch(() => undefined);
      const detectorSession = this.detectorSession;
      const landmarkSession = this.landmarkSession;
      this.detectorSession = null;
      this.landmarkSession = null;
      await detectorSession?.release();
      await landmarkSession?.release();
    })();

    this.disposePromise = disposal;
    return disposal;
  }

  private async loadSessions() {
    const detectorSession = await this.createSession(
      this.config.modelPath,
      this.config.externalDataPath
    );
    if (this.disposed) {
      await detectorSession.release();
      throw new Error(`${this.name} was disposed while loading`);
    }

    let landmarkSession: ort.InferenceSession | null = null;
    try {
      if (this.config.landmarkModelPath) {
        landmarkSession = await this.createSession(
          this.config.landmarkModelPath,
          this.config.landmarkDataPath
        );
      }
      if (this.disposed) {
        await detectorSession.release();
        await landmarkSession?.release();
        throw new Error(`${this.name} was disposed while loading`);
      }
      this.detectorSession = detectorSession;
      this.landmarkSession = landmarkSession;
    } catch (error) {
      if (this.detectorSession !== detectorSession) {
        await detectorSession.release();
      }
      await landmarkSession?.release();
      throw error;
    }
  }

  private async createSession(
    modelPath: string,
    externalDataPath?: string
  ): Promise<ort.InferenceSession> {
    if (!externalDataPath) {
      return ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
      });
    }

    const response = await fetch(externalDataPath);
    if (!response.ok) {
      throw new Error(`Could not load model weights: ${externalDataPath} (${response.status})`);
    }
    const data = new Uint8Array(await response.arrayBuffer());
    const externalDataName = externalDataPath.slice(externalDataPath.lastIndexOf("/") + 1);
    return ort.InferenceSession.create(modelPath, {
      executionProviders: ["wasm"],
      externalData: [{ path: externalDataName, data }],
    });
  }

  private async detectFrame(image: ImageData) {
    if (image.width <= 0 || image.height <= 0) {
      throw new Error("Image dimensions must be positive");
    }
    const detectorSession = this.detectorSession;
    if (!detectorSession) throw new Error(`${this.name} must be loaded before detect()`);

    const timings: DetectorTimings = {
      preprocessingMs: 0,
      inferenceMs: 0,
      decodingMs: 0,
    };
    const preprocessStartedAt = performance.now();
    const transform = this.prepareInput(image, this.config.inputSize, this.config.layout);
    const detectorTensor = this.createTensor(
      this.detectorInputData!,
      this.config.inputSize,
      this.config.layout
    );
    timings.preprocessingMs += performance.now() - preprocessStartedAt;

    let detectorOutputs: Awaited<ReturnType<ort.InferenceSession["run"]>> | null = null;
    let faces: CandidateFace[];
    try {
      const inferenceStartedAt = performance.now();
      detectorOutputs = await detectorSession.run({
        [detectorSession.inputNames[0]]: detectorTensor,
      });
      timings.inferenceMs += performance.now() - inferenceStartedAt;

      const decodingStartedAt = performance.now();
      faces = this.decodeDetections(detectorSession, detectorOutputs, image, transform);
      timings.decodingMs += performance.now() - decodingStartedAt;
    } finally {
      if (detectorOutputs) disposeOutputs(detectorOutputs);
      detectorTensor.dispose();
    }

    const landmarkSession = this.landmarkSession;
    if (landmarkSession) {
      const landmarkedFaces: CandidateFace[] = [];
      for (const face of faces) {
        const crop = faceCrop(face);
        const landmarkPreprocessStartedAt = performance.now();
        this.prepareLandmarkInput(image, crop);
        const landmarkTensor = this.createTensor(
          this.landmarkInputData!,
          LANDMARK_INPUT_SIZE,
          "nchw"
        );
        timings.preprocessingMs += performance.now() - landmarkPreprocessStartedAt;

        let landmarkOutputs: Awaited<ReturnType<ort.InferenceSession["run"]>> | null = null;
        try {
          const inferenceStartedAt = performance.now();
          landmarkOutputs = await landmarkSession.run({
            [landmarkSession.inputNames[0]]: landmarkTensor,
          });
          timings.inferenceMs += performance.now() - inferenceStartedAt;

          const decodingStartedAt = performance.now();
          const score = tensorData(landmarkOutputs, "scores")[0];
          if (score >= LANDMARK_THRESHOLD) {
            face.landmarks = decodeLandmarks(
              tensorData(landmarkOutputs, "landmarks"),
              crop
            );
            landmarkedFaces.push(face);
          }
          timings.decodingMs += performance.now() - decodingStartedAt;
        } finally {
          if (landmarkOutputs) disposeOutputs(landmarkOutputs);
          landmarkTensor.dispose();
        }
      }
      faces = landmarkedFaces;
    }

    return { faces, timings };
  }

  private prepareInput(
    image: ImageData,
    size: number,
    layout: ModelLayout
  ): FrameTransform {
    const sourceCanvas = this.sourceCanvas ??= document.createElement("canvas");
    if (sourceCanvas.width !== image.width) sourceCanvas.width = image.width;
    if (sourceCanvas.height !== image.height) sourceCanvas.height = image.height;
    const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
    if (!sourceContext) throw new Error(`Could not initialize ${this.name} source canvas`);
    sourceContext.putImageData(image, 0, 0);

    const canvas = this.modelCanvas ??= document.createElement("canvas");
    if (canvas.width !== size) canvas.width = size;
    if (canvas.height !== size) canvas.height = size;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error(`Could not initialize ${this.name} model canvas`);

    const scale = Math.min(size / image.width, size / image.height);
    const width = image.width * scale;
    const height = image.height * scale;
    const offsetX = (size - width) / 2;
    const offsetY = (size - height) / 2;
    context.fillStyle = "black";
    context.fillRect(0, 0, size, size);
    context.drawImage(sourceCanvas, offsetX, offsetY, width, height);

    const data = context.getImageData(0, 0, size, size).data;
    const values = this.detectorInputData ?? new Float32Array(size * size * 3);
    const pixels = size * size;
    for (let i = 0; i < pixels; i++) {
      const rgba = i * 4;
      const red = data[rgba] / 255;
      const green = data[rgba + 1] / 255;
      const blue = data[rgba + 2] / 255;
      if (layout === "nhwc") {
        const offset = i * 3;
        values[offset] = red;
        values[offset + 1] = green;
        values[offset + 2] = blue;
      } else {
        values[i] = red;
        values[pixels + i] = green;
        values[2 * pixels + i] = blue;
      }
    }
    this.detectorInputData = values;
    return { scale, offsetX, offsetY };
  }

  private prepareLandmarkInput(image: ImageData, crop: CropRect) {
    const size = LANDMARK_INPUT_SIZE;
    const sourceCanvas = this.sourceCanvas ??= document.createElement("canvas");
    if (sourceCanvas.width !== image.width) sourceCanvas.width = image.width;
    if (sourceCanvas.height !== image.height) sourceCanvas.height = image.height;
    const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
    if (!sourceContext) throw new Error(`Could not initialize ${this.name} source canvas`);
    sourceContext.putImageData(image, 0, 0);

    const canvas = this.modelCanvas ??= document.createElement("canvas");
    if (canvas.width !== size) canvas.width = size;
    if (canvas.height !== size) canvas.height = size;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error(`Could not initialize ${this.name} landmark canvas`);
    context.fillStyle = "black";
    context.fillRect(0, 0, size, size);
    context.drawImage(sourceCanvas, crop.x, crop.y, crop.width, crop.height, 0, 0, size, size);

    const data = context.getImageData(0, 0, size, size).data;
    const values = this.landmarkInputData ??= new Float32Array(size * size * 3);
    const pixels = size * size;
    for (let i = 0; i < pixels; i++) {
      const rgba = i * 4;
      values[i] = data[rgba] / 255;
      values[pixels + i] = data[rgba + 1] / 255;
      values[2 * pixels + i] = data[rgba + 2] / 255;
    }
  }

  private createTensor(
    data: Float32Array,
    size: number,
    layout: ModelLayout
  ): ort.Tensor {
    const dimensions = layout === "nhwc" ? [1, size, size, 3] : [1, 3, size, size];
    return new ort.Tensor("float32", data, dimensions);
  }

  private decodeDetections(
    session: ort.InferenceSession,
    outputs: Awaited<ReturnType<ort.InferenceSession["run"]>>,
    image: ImageData,
    transform: FrameTransform
  ): CandidateFace[] {
    const outputNames = session.outputNames;
    const getOutput = (name: string) => tensorData(outputs, name);
    let boxData: Float32Array;
    let scoreData: Float32Array;

    if (outputNames.includes("regressors") && outputNames.includes("classificators")) {
      boxData = getOutput("regressors");
      scoreData = getOutput("classificators");
    } else {
      boxData = concatFloat32(getOutput("box_coords_1"), getOutput("box_coords_2"));
      scoreData = concatFloat32(getOutput("box_scores_1"), getOutput("box_scores_2"));
    }

    if (boxData.length !== ANCHORS.length * 16 || scoreData.length !== ANCHORS.length) {
      throw new Error(`${this.name} returned unexpected detection tensor dimensions`);
    }

    const candidates: CandidateFace[] = [];
    const size = this.config.inputSize;
    for (let index = 0; index < ANCHORS.length; index++) {
      const score = sigmoid(scoreData[index]);
      if (score < (this.config.confidenceThreshold ?? FACE_THRESHOLD)) continue;

      const anchor = ANCHORS[index];
      const offset = index * 16;
      const centerX = boxData[offset] / size + anchor.x;
      const centerY = boxData[offset + 1] / size + anchor.y;
      const width = boxData[offset + 2] / size;
      const height = boxData[offset + 3] / size;
      const x = this.toImageX((centerX - width / 2) * size, transform);
      const y = this.toImageY((centerY - height / 2) * size, transform);
      const right = this.toImageX((centerX + width / 2) * size, transform);
      const bottom = this.toImageY((centerY + height / 2) * size, transform);
      const leftClamped = clamp(x, 0, image.width);
      const topClamped = clamp(y, 0, image.height);
      const rightClamped = clamp(right, 0, image.width);
      const bottomClamped = clamp(bottom, 0, image.height);
      if (rightClamped <= leftClamped || bottomClamped <= topClamped) continue;

      const landmarks = [];
      for (let point = 0; point < 6; point++) {
        const pointX = (boxData[offset + 4 + point * 2] / size + anchor.x) * size;
        const pointY = (boxData[offset + 5 + point * 2] / size + anchor.y) * size;
        landmarks.push({
          x: clamp(this.toImageX(pointX, transform), 0, image.width),
          y: clamp(this.toImageY(pointY, transform), 0, image.height),
        });
      }
      candidates.push({
        x: leftClamped,
        y: topClamped,
        width: rightClamped - leftClamped,
        height: bottomClamped - topClamped,
        confidence: score,
        landmarks,
      });
    }

    return nonMaximumSuppression(candidates);
  }

  private toImageX(modelX: number, transform: FrameTransform) {
    return (modelX - transform.offsetX) / transform.scale;
  }

  private toImageY(modelY: number, transform: FrameTransform) {
    return (modelY - transform.offsetY) / transform.scale;
  }
}

export class BlazeFaceShortRangeDetector extends BlazeFaceOnnxDetector {
  constructor() {
    super({
      id: "blazeface",
      name: "MediaPipe BlazeFace Short Range",
      modelPath: "/models/blazeface.onnx",
      inputSize: 128,
      layout: "nhwc",
      confidenceThreshold: 0.6,
    });
  }
}

export class MediaPipeLandmarkDetector extends BlazeFaceOnnxDetector {
  constructor() {
    super({
      id: "mediapipe",
      name: "MediaPipe Face Detection + Landmarks",
      modelPath: "/models/mediapipe/face_detector.onnx",
      inputSize: 256,
      layout: "nchw",
      externalDataPath: "/models/mediapipe/face_detector.data",
      landmarkModelPath: "/models/mediapipe/face_landmark_detector.onnx",
      landmarkDataPath: "/models/mediapipe/face_landmark_detector.data",
    });
  }
}

type CropRect = { x: number; y: number; width: number; height: number };

function createAnchors() {
  const anchors: { x: number; y: number }[] = [];
  for (const stride of [8, 16, 16, 16]) {
    const featureSize = 128 / stride;
    for (let y = 0; y < featureSize; y++) {
      for (let x = 0; x < featureSize; x++) {
        for (let anchor = 0; anchor < 2; anchor++) {
          anchors.push({
            x: (x + 0.5) / featureSize,
            y: (y + 0.5) / featureSize,
          });
        }
      }
    }
  }
  return anchors;
}

function tensorData(
  outputs: Awaited<ReturnType<ort.InferenceSession["run"]>>,
  name: string
): Float32Array {
  const tensor = outputs[name];
  if (!tensor || !(tensor.data instanceof Float32Array)) {
    throw new Error(`Model output "${name}" is missing or is not float32`);
  }
  return tensor.data;
}

function concatFloat32(first: Float32Array, second: Float32Array) {
  const combined = new Float32Array(first.length + second.length);
  combined.set(first);
  combined.set(second, first.length);
  return combined;
}

function disposeOutputs(
  outputs: Awaited<ReturnType<ort.InferenceSession["run"]>>
) {
  Object.values(outputs).forEach((output) => output.dispose());
}

function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-value));
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function calculateIoU(a: DetectedFace, b: DetectedFace) {
  const intersectionWidth = Math.max(
    0,
    Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x)
  );
  const intersectionHeight = Math.max(
    0,
    Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y)
  );
  const intersection = intersectionWidth * intersectionHeight;
  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function nonMaximumSuppression(faces: CandidateFace[]) {
  const selected: CandidateFace[] = [];
  for (const face of faces.sort((a, b) => b.confidence - a.confidence)) {
    if (selected.every((existing) => calculateIoU(existing, face) < NMS_THRESHOLD)) {
      selected.push(face);
      if (selected.length === MAX_FACES) break;
    }
  }
  return selected;
}

function faceCrop(face: DetectedFace): CropRect {
  const size = Math.max(face.width, face.height) * 1.5;
  return {
    x: face.x + face.width / 2 - size / 2,
    y: face.y + face.height / 2 - size / 2,
    width: size,
    height: size,
  };
}

function decodeLandmarks(values: Float32Array, crop: CropRect) {
  if (values.length !== 468 * 3) {
    throw new Error("MediaPipe landmark model returned unexpected tensor dimensions");
  }
  const landmarks = [];
  for (let index = 0; index < 468; index++) {
    const x = values[index * 3];
    const y = values[index * 3 + 1];
    landmarks.push({
      x: crop.x + x * crop.width,
      y: crop.y + y * crop.height,
    });
  }
  return landmarks;
}
