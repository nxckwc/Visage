import type { DetectedFace } from "./detectors/types";
import { alignFaceImage } from "./faceAlignment";

export type FaceQuality = { acceptable: boolean; issues: string[]; alignedImage?: HTMLCanvasElement; alignmentMs: number; qualityMs: number };

/** Conservative capture heuristics; these require dataset calibration before authentication use. */
export function checkFaceQuality(image: ImageData, face: DetectedFace, faceCount: number): FaceQuality {
  const qualityStartedAt = performance.now();
  const issues: string[] = [];
  if (faceCount !== 1) issues.push(faceCount === 0 ? "Center one face in view" : "Only one face should be in view");
  if (face.confidence < 0.6) issues.push("Face detection confidence is low; face the camera");
  if (face.width < 90 || face.height < 90) issues.push("Move closer to the camera");
  let alignedImage: HTMLCanvasElement | undefined;
  let alignmentMs = 0;
  try {
    const alignmentStartedAt = performance.now();
    alignedImage = alignFaceImage(image, face);
    alignmentMs = performance.now() - alignmentStartedAt;
    const landmarks = face.landmarks.length >= 264
      ? [face.landmarks[33], face.landmarks[263], face.landmarks[1], face.landmarks[61], face.landmarks[291]]
      : face.landmarks.slice(0, 5);
    if (landmarks.length < 5 || landmarks.some((point) => !point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || point.x < 0 || point.y < 0 || point.x >= image.width || point.y >= image.height)) {
      issues.push("Facial landmarks are incomplete; face the camera clearly");
    } else {
      const [leftEye, rightEye, nose, leftMouth, rightMouth] = landmarks;
      const eyeDistance = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
      const mouthDistance = Math.hypot(rightMouth.x - leftMouth.x, rightMouth.y - leftMouth.y);
      const eyeMidX = (leftEye.x + rightEye.x) / 2;
      const eyeMidY = (leftEye.y + rightEye.y) / 2;
      const mouthMidY = (leftMouth.y + rightMouth.y) / 2;
      if (eyeDistance < face.width * 0.12 || mouthDistance < face.width * 0.1 || mouthMidY <= eyeMidY || nose.y <= eyeMidY || nose.y >= mouthMidY) {
        issues.push("Landmark layout is unclear; face camera and uncover eyes, nose, and mouth");
      } else if (Math.abs(nose.x - eyeMidX) / eyeDistance > 0.55) {
        issues.push("Turn toward the camera");
      } else {
        const nosePosition = (nose.y - eyeMidY) / (mouthMidY - eyeMidY);
        if (nosePosition < 0.15 || nosePosition > 0.85) issues.push("Adjust head tilt toward the camera");
      }
      const roll = Math.abs(Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x));
      if (roll > Math.PI / 2) issues.push("Turn your head upright");
    }
    const ctx = alignedImage.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Could not inspect aligned face");
    const pixels = ctx.getImageData(0, 0, 112, 112).data;
    const gray = new Float32Array(112 * 112);
    let total = 0, clipped = 0;
    for (let i = 0; i < gray.length; i++) {
      const value = 0.299 * pixels[i * 4] + 0.587 * pixels[i * 4 + 1] + 0.114 * pixels[i * 4 + 2];
      gray[i] = value; total += value;
      if (value < 8 || value > 247) clipped++;
    }
    const mean = total / gray.length;
    if (mean < 35) issues.push("Improve lighting; the image is too dark");
    else if (mean > 220) issues.push("Reduce strong light; the image is overexposed");
    if (clipped / gray.length > 0.4) issues.push("Improve lighting to recover facial detail");
    let lapTotal = 0, lapSquared = 0, samples = 0;
    for (let y = 1; y < 111; y++) for (let x = 1; x < 111; x++) {
      const i = y * 112 + x;
      const value = gray[i - 112] + gray[i - 1] - 4 * gray[i] + gray[i + 1] + gray[i + 112];
      lapTotal += value; lapSquared += value * value; samples++;
    }
    const lapMean = lapTotal / samples, lapVariance = lapSquared / samples - lapMean * lapMean;
    if (lapVariance < 18) issues.push("Hold still; the face is blurry");
  } catch (error) {
    issues.push(error instanceof Error ? error.message : "Face alignment failed");
  }
  return { acceptable: issues.length === 0, issues: [...new Set(issues)], alignedImage, alignmentMs, qualityMs: Math.max(0, performance.now() - qualityStartedAt - alignmentMs) };
}
