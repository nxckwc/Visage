import type { DetectedFace } from "./detectors/types";

const OUTPUT_SIZE = 112;
const REFERENCE_POINTS = [
  { x: 38.2946, y: 51.6963 },
  { x: 73.5318, y: 51.5014 },
  { x: 56.0252, y: 71.7366 },
  { x: 41.5493, y: 92.3655 },
  { x: 70.7299, y: 92.2041 },
];

function fivePoints(face: DetectedFace): { x: number; y: number }[] {
  const landmarks = face.landmarks.length >= 264
    ? [face.landmarks[33], face.landmarks[263], face.landmarks[1], face.landmarks[61], face.landmarks[291]]
    : face.landmarks.slice(0, 5);
  if (landmarks.length < 5 || landmarks.some((point) => !point || !Number.isFinite(point.x) || !Number.isFinite(point.y))) {
    throw new Error("Face alignment requires five valid eye, nose, and mouth landmarks");
  }
  return landmarks;
}

function estimateSimilarity(source: { x: number; y: number }[], target: typeof REFERENCE_POINTS) {
  const mean = (points: { x: number; y: number }[]) => ({
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
  });
  const sourceMean = mean(source), targetMean = mean(target);
  let a = 0, b = 0, denominator = 0;
  for (let i = 0; i < source.length; i++) {
    const x = source[i].x - sourceMean.x, y = source[i].y - sourceMean.y;
    const u = target[i].x - targetMean.x, v = target[i].y - targetMean.y;
    a += x * u + y * v;
    b += x * v - y * u;
    denominator += x * x + y * y;
  }
  if (denominator < 1e-6) throw new Error("Face landmarks are too close together to align");
  const magnitude = Math.hypot(a, b);
  if (!Number.isFinite(magnitude) || magnitude < 1e-6) throw new Error("Face landmarks cannot be aligned");
  const scale = magnitude / denominator, cosine = a / magnitude, sine = b / magnitude;
  const m11 = scale * cosine, m12 = scale * sine, m21 = -scale * sine, m22 = scale * cosine;
  return {
    a: m11, b: m12, c: m21, d: m22,
    e: targetMean.x - (m11 * sourceMean.x + m21 * sourceMean.y),
    f: targetMean.y - (m12 * sourceMean.x + m22 * sourceMean.y),
  };
}

/** Warp the source face directly into the model's five-point 112x112 template. */
export function alignFaceImage(image: ImageData, face: DetectedFace): HTMLCanvasElement {
  if (![face.x, face.y, face.width, face.height].every(Number.isFinite) || face.width <= 0 || face.height <= 0) {
    throw new Error("Face detector returned invalid bounds");
  }
  if (!image.width || !image.height) throw new Error("Cannot align an empty frame");
  const transform = estimateSimilarity(fivePoints(face), REFERENCE_POINTS);
  const source = document.createElement("canvas");
  source.width = image.width;
  source.height = image.height;
  const sourceContext = source.getContext("2d", { willReadFrequently: true });
  if (!sourceContext) throw new Error("Could not create face alignment source");
  sourceContext.putImageData(image, 0, 0);

  const output = document.createElement("canvas");
  output.width = OUTPUT_SIZE;
  output.height = OUTPUT_SIZE;
  const context = output.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Could not create aligned face image");
  context.fillStyle = "black";
  context.fillRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
  context.setTransform(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f);
  context.drawImage(source, 0, 0);
  return output;
}
