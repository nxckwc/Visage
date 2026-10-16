import * as ort from "onnxruntime-web";
import type { DetectedFace } from "./detectors/types";
import { alignFaceImage } from "./faceAlignment";

export const GHOSTFACENET_MODEL_ID = "ghostfacenet_v1" as const;
export const GHOSTFACENET_PREPROCESSING_ID = "arcface-5point-112-v1" as const;
export const GHOSTFACENET_EMBEDDING_SIZE = 512;
export const ARCFACE_RESNET100_MODEL_ID = "arcface_resnet100_v1" as const;
export const ARCFACE_RESNET100_PREPROCESSING_ID = "arcface-resnet100-bgr-nchw-112-v1" as const;
export type RecognitionModelId = typeof GHOSTFACENET_MODEL_ID | typeof ARCFACE_RESNET100_MODEL_ID;
export const RECOGNITION_MODELS: Record<RecognitionModelId, { label: string; preprocessing: string; modelUrl: string }> = {
  [GHOSTFACENET_MODEL_ID]: { label: "GhostFaceNet V1", preprocessing: GHOSTFACENET_PREPROCESSING_ID, modelUrl: "/models/ghostfacenet_v1.onnx" },
  [ARCFACE_RESNET100_MODEL_ID]: { label: "ArcFace ResNet100 (ONNX Model Zoo)", preprocessing: ARCFACE_RESNET100_PREPROCESSING_ID, modelUrl: "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx" },
};
const IMAGE_SIZE = 112;

export async function createGhostFaceNetSession(preference: "auto" | "wasm" | "webgpu" = "auto") {
  const useWebGpu = preference !== "wasm" && typeof navigator !== "undefined" && "gpu" in navigator;
  const executionProviders = useWebGpu ? ["webgpu", "wasm"] : ["wasm"];
  const session = await ort.InferenceSession.create("/models/ghostfacenet_v1.onnx", { executionProviders });
  return { session, backend: useWebGpu ? "WebGPU preferred with WASM fallback" : "WASM" };
}

export async function createRecognitionSession(modelId: RecognitionModelId, preference: "auto" | "wasm" | "webgpu" = "auto") {
  if (modelId === GHOSTFACENET_MODEL_ID) return createGhostFaceNetSession(preference);
  const url = RECOGNITION_MODELS[modelId].modelUrl;
  const useWebGpu = preference !== "wasm" && typeof navigator !== "undefined" && "gpu" in navigator;
  const executionProviders = useWebGpu ? ["webgpu", "wasm"] : ["wasm"];
  const session = await ort.InferenceSession.create(url, { executionProviders });
  return { session, backend: useWebGpu ? "WebGPU preferred with WASM fallback" : "WASM" };
}

function readInputPixels(image: HTMLCanvasElement) {
  const pixels = image.getContext("2d", { willReadFrequently: true })?.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  if (!pixels) throw new Error("Could not read aligned face pixels");
  const values = new Float32Array(IMAGE_SIZE * IMAGE_SIZE * 3);
  for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
    values[i * 3] = pixels[i * 4] / 127.5 - 1;
    values[i * 3 + 1] = pixels[i * 4 + 1] / 127.5 - 1;
    values[i * 3 + 2] = pixels[i * 4 + 2] / 127.5 - 1;
  }
  return values;
}

export async function embedAlignedFace(session: ort.InferenceSession, alignedImage: HTMLCanvasElement, modelId: RecognitionModelId = GHOSTFACENET_MODEL_ID) {
  const dims = session.inputMetadata[0] && "shape" in session.inputMetadata[0] ? session.inputMetadata[0].shape : undefined;
  const isGhost = modelId === GHOSTFACENET_MODEL_ID;
  const validShape = isGhost
    ? dims?.length === 4 && dims[1] === IMAGE_SIZE && dims[2] === IMAGE_SIZE && dims[3] === 3
    : dims?.length === 4 && dims[1] === 3 && dims[2] === IMAGE_SIZE && dims[3] === IMAGE_SIZE;
  if (!validShape) {
    throw new Error(`${RECOGNITION_MODELS[modelId].label} has an unsupported input shape [${dims?.join(", ") ?? "unknown"}]`);
  }
  if (alignedImage.width !== IMAGE_SIZE || alignedImage.height !== IMAGE_SIZE) {
    throw new Error(`Aligned face must be ${IMAGE_SIZE}x${IMAGE_SIZE}`);
  }
  let pixels = readInputPixels(alignedImage);
  let tensor: ort.Tensor;
  if (isGhost) tensor = new ort.Tensor("float32", pixels, [1, IMAGE_SIZE, IMAGE_SIZE, 3]);
  else {
    const chw = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);
    for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
      // The ONNX Model Zoo ResNet100 export consumes BGR, NCHW values in [-1, 1].
      chw[i] = pixels[i * 3 + 2];
      chw[IMAGE_SIZE * IMAGE_SIZE + i] = pixels[i * 3 + 1];
      chw[2 * IMAGE_SIZE * IMAGE_SIZE + i] = pixels[i * 3];
    }
    pixels = chw;
    tensor = new ort.Tensor("float32", pixels, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  }
  try {
    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    try {
      const output = Object.values(outputs)[0];
      if (!output || output.dims.length !== 2 || output.dims[0] !== 1 || output.dims[1] !== GHOSTFACENET_EMBEDDING_SIZE) {
        throw new Error(`${RECOGNITION_MODELS[modelId].label} expects [1,${GHOSTFACENET_EMBEDDING_SIZE}] output; got [${output?.dims.join(", ") ?? "no output"}]`);
      }
      if (!(output.data instanceof Float32Array)) throw new Error("GhostFaceNet returned an unsupported embedding format");
      const vector = Array.from(output.data, Number);
      if (vector.some((value) => !Number.isFinite(value))) throw new Error("GhostFaceNet returned NaN or infinite embedding values");
      const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
      if (!norm || !Number.isFinite(norm)) throw new Error("GhostFaceNet returned a zero-norm or invalid embedding");
      return vector.map((value) => value / norm);
    } finally { Object.values(outputs).forEach((output) => output.dispose()); }
  } finally { tensor.dispose(); }
}

export async function extractFaceEmbedding(session: ort.InferenceSession, image: ImageData, face: DetectedFace, modelId: RecognitionModelId = GHOSTFACENET_MODEL_ID) {
  const alignmentStartedAt = performance.now();
  const alignedImage = alignFaceImage(image, face);
  const alignmentMs = performance.now() - alignmentStartedAt;
  const inferenceStartedAt = performance.now();
  const vector = await embedAlignedFace(session, alignedImage, modelId);
  return { embedding: vector, alignedImage, alignmentMs, inferenceMs: performance.now() - inferenceStartedAt };
}

export function cosineSimilarity(a: number[], b: number[]) {
  if (a.length !== GHOSTFACENET_EMBEDDING_SIZE || b.length !== GHOSTFACENET_EMBEDDING_SIZE ||
      a.some((value) => !Number.isFinite(value)) || b.some((value) => !Number.isFinite(value))) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i]; }
  return normA && normB ? dot / Math.sqrt(normA * normB) : 0;
}
