import { describe, expect, it } from "vitest";
import type { DetectedFace } from "../src/detectors/types";
import { smoothFaces } from "../src/detectionOverlay";

function face(x: number, confidence = 0.9): DetectedFace {
  return {
    x,
    y: 20,
    width: 100,
    height: 120,
    confidence,
    landmarks: [
      { x: x + 25, y: 50 },
      { x: x + 70, y: 50 },
      { x: x + 48, y: 75 },
      { x: x + 32, y: 100 },
      { x: x + 64, y: 100 },
    ],
  };
}

describe("smoothFaces", () => {
  it("dampens frame-to-frame movement without changing confidence", () => {
    const [result] = smoothFaces([face(0)], [face(10, 0.8)]);

    expect(result.x).toBeCloseTo(4.2);
    expect(result.landmarks[0].x).toBeCloseTo(29.2);
    expect(result.confidence).toBe(0.8);
  });

  it("keeps separate faces matched to their nearest prior geometry", () => {
    const previous = [face(0), face(240)];
    const result = smoothFaces(previous, [face(250), face(10)]);

    expect(result.map(({ x }) => x)).toEqual([244.2, 4.2]);
  });

  it("does not keep stale detections when a face disappears", () => {
    expect(smoothFaces([face(0)], [])).toEqual([]);
  });

  it("does not blend a detection with an unrelated face", () => {
    const [result] = smoothFaces([face(0)], [face(400)]);

    expect(result).toEqual(face(400));
  });
});
