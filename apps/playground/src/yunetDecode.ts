
import type * as ort from "onnxruntime-web";

export type Face = {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  landmarks: Array<{ x: number; y: number }>;
};

type Transform = {
  scale: number;
  offsetX: number;
  offsetY: number;
  sourceWidth: number;
  sourceHeight: number;
};

function iou(a: Face, b: Face): number {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);

  const intersection =
    Math.max(0, right - left) * Math.max(0, bottom - top);

  const union =
    a.width * a.height + b.width * b.height - intersection;

  return union > 0 ? intersection / union : 0;
}

export function decodeYuNet(
  outputs: Record<string, ort.Tensor>,
  transform: Transform,
  confidenceThreshold = 0.6,
  nmsThreshold = 0.3
): Face[] {
  const candidates: Face[] = [];

  const toOriginalX = (x: number) =>
    (x - transform.offsetX) / transform.scale;

  const toOriginalY = (y: number) =>
    (y - transform.offsetY) / transform.scale;

  for (const stride of [8, 16, 32]) {
    const cls = outputs[`cls_${stride}`].data as Float32Array;
    const obj = outputs[`obj_${stride}`].data as Float32Array;
    const bbox = outputs[`bbox_${stride}`].data as Float32Array;
    const kps = outputs[`kps_${stride}`].data as Float32Array;

    const featureWidth = 640 / stride;

    for (let i = 0; i < cls.length; i++) {
      const score = Math.sqrt(
        Math.max(0, Math.min(1, cls[i])) *
        Math.max(0, Math.min(1, obj[i]))
      );

      if (score < confidenceThreshold) continue;

      const col = i % featureWidth;
      const row = Math.floor(i / featureWidth);

      const cx = (col + bbox[i * 4]) * stride;
      const cy = (row + bbox[i * 4 + 1]) * stride;
      const width = Math.exp(bbox[i * 4 + 2]) * stride;
      const height = Math.exp(bbox[i * 4 + 3]) * stride;

      const x1 = toOriginalX(cx - width / 2);
      const y1 = toOriginalY(cy - height / 2);
      const x2 = toOriginalX(cx + width / 2);
      const y2 = toOriginalY(cy + height / 2);

      const x = Math.max(0, Math.min(transform.sourceWidth, x1));
      const y = Math.max(0, Math.min(transform.sourceHeight, y1));
      const right = Math.max(0, Math.min(transform.sourceWidth, x2));
      const bottom = Math.max(0, Math.min(transform.sourceHeight, y2));

      if (right <= x || bottom <= y) continue;

      const landmarks = Array.from({ length: 5 }, (_, k) => ({
        x: toOriginalX(
          (col + kps[i * 10 + k * 2]) * stride
        ),
        y: toOriginalY(
          (row + kps[i * 10 + k * 2 + 1]) * stride
        ),
      }));

      candidates.push({
        x,
        y,
        width: right - x,
        height: bottom - y,
        confidence: score,
        landmarks,
      });
    }
  }

  // Non-maximum suppression: remove overlapping detections.
  candidates.sort((a, b) => b.confidence - a.confidence);

  const faces: Face[] = [];

  for (const candidate of candidates) {
    if (faces.every((face) => iou(face, candidate) < nmsThreshold)) {
      faces.push(candidate);
    }

    if (faces.length >= 100) break;
  }

  return faces;
}
