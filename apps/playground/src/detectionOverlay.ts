import type { DetectedFace } from "./detectors/types";

const MATCH_IOU_THRESHOLD = 0.12;
const CURRENT_FRAME_WEIGHT = 0.42;

function intersectionOverUnion(a: DetectedFace, b: DetectedFace): number {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function blend(previous: number, current: number): number {
  return previous + (current - previous) * CURRENT_FRAME_WEIGHT;
}

function smoothFace(previous: DetectedFace, current: DetectedFace): DetectedFace {
  return {
    x: blend(previous.x, current.x),
    y: blend(previous.y, current.y),
    width: blend(previous.width, current.width),
    height: blend(previous.height, current.height),
    confidence: current.confidence,
    landmarks: current.landmarks.map((point, index) => {
      const oldPoint = previous.landmarks[index];
      return oldPoint
        ? { x: blend(oldPoint.x, point.x), y: blend(oldPoint.y, point.y) }
        : point;
    }),
  };
}

export function smoothFaces(
  previousFaces: DetectedFace[],
  currentFaces: DetectedFace[]
): DetectedFace[] {
  const matchedPrevious = new Set<number>();

  return currentFaces.map((face) => {
    let bestIndex = -1;
    let bestIou = MATCH_IOU_THRESHOLD;

    for (let index = 0; index < previousFaces.length; index++) {
      if (matchedPrevious.has(index)) continue;
      const overlap = intersectionOverUnion(previousFaces[index], face);
      if (overlap > bestIou) {
        bestIou = overlap;
        bestIndex = index;
      }
    }

    if (bestIndex === -1) return face;
    matchedPrevious.add(bestIndex);
    return smoothFace(previousFaces[bestIndex], face);
  });
}

function roundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
): void {
  context.beginPath();
  context.roundRect(x, y, width, height, radius);
}

export function drawFaces(
  context: CanvasRenderingContext2D,
  faces: DetectedFace[]
): void {
  const scale = Math.max(0.75, Math.min(context.canvas.width, context.canvas.height) / 480);
  const lineWidth = 2 * scale;
  const cornerLength = 14 * scale;
  const pointRadius = 3 * scale;
  const fontSize = 13 * scale;

  context.save();
  context.lineWidth = lineWidth;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.font = `600 ${fontSize}px ui-monospace, monospace`;

  for (const face of faces) {
    const { x, y, width, height } = face;
    const right = x + width;
    const bottom = y + height;

    context.strokeStyle = "rgba(74, 241, 190, 0.96)";
    context.shadowColor = "rgba(36, 220, 170, 0.4)";
    context.shadowBlur = 8 * scale;
    roundedRect(context, x, y, width, height, 8 * scale);
    context.stroke();
    context.shadowBlur = 0;

    context.beginPath();
    for (const [cornerX, cornerY, dx, dy] of [
      [x, y, 1, 1],
      [right, y, -1, 1],
      [x, bottom, 1, -1],
      [right, bottom, -1, -1],
    ]) {
      context.moveTo(cornerX, cornerY + dy * cornerLength);
      context.lineTo(cornerX, cornerY);
      context.lineTo(cornerX + dx * cornerLength, cornerY);
    }
    context.strokeStyle = "#65ffd0";
    context.lineWidth = 3 * scale;
    context.stroke();
    context.lineWidth = lineWidth;

    const [leftEye, rightEye, nose, leftMouth, rightMouth] = face.landmarks;
    if (leftEye && rightEye && nose && leftMouth && rightMouth) {
      context.beginPath();
      context.moveTo(leftEye.x, leftEye.y);
      context.lineTo(rightEye.x, rightEye.y);
      context.moveTo(leftEye.x, leftEye.y);
      context.lineTo(nose.x, nose.y);
      context.lineTo(rightEye.x, rightEye.y);
      context.moveTo(nose.x, nose.y);
      context.lineTo((leftMouth.x + rightMouth.x) / 2, (leftMouth.y + rightMouth.y) / 2);
      context.moveTo(leftMouth.x, leftMouth.y);
      context.lineTo(rightMouth.x, rightMouth.y);
      context.strokeStyle = "rgba(102, 231, 255, 0.92)";
      context.lineWidth = 1.5 * scale;
      context.stroke();
    }

    for (const point of face.landmarks) {
      context.beginPath();
      context.arc(point.x, point.y, pointRadius + 1.5 * scale, 0, Math.PI * 2);
      context.fillStyle = "rgba(18, 26, 37, 0.78)";
      context.fill();
      context.beginPath();
      context.arc(point.x, point.y, pointRadius, 0, Math.PI * 2);
      context.fillStyle = "#78efff";
      context.fill();
    }

    const label = `${Math.round(face.confidence * 100)}%`;
    const labelWidth = context.measureText(label).width + 14 * scale;
    const labelHeight = 22 * scale;
    const labelX = Math.max(0, Math.min(x, context.canvas.width - labelWidth));
    const labelY = y >= labelHeight + 4 * scale
      ? y - labelHeight - 4 * scale
      : Math.min(bottom + 4 * scale, context.canvas.height - labelHeight);
    roundedRect(context, labelX, labelY, labelWidth, labelHeight, 6 * scale);
    context.fillStyle = "rgba(15, 25, 31, 0.88)";
    context.fill();
    context.fillStyle = "#8dffdc";
    context.textBaseline = "middle";
    context.fillText(label, labelX + 7 * scale, labelY + labelHeight / 2);
  }

  context.restore();
}
