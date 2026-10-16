import { afterEach, describe, expect, it, vi } from "vitest";
import type { DetectedFace } from "../src/detectors/types";
import { alignFaceImage } from "../src/faceAlignment";

function canvasContext() {
  return {
    fillStyle: "",
    putImageData: vi.fn(),
    fillRect: vi.fn(),
    setTransform: vi.fn(),
    drawImage: vi.fn(),
  };
}

function mockCanvas() {
  const contexts = [canvasContext(), canvasContext()];
  let index = 0;
  vi.stubGlobal("document", {
    createElement: vi.fn(() => ({
      width: 0,
      height: 0,
      getContext: vi.fn(() => contexts[index++]),
    })),
  });
  return contexts;
}

const template = [
  { x: 38.2946, y: 51.6963 },
  { x: 73.5318, y: 51.5014 },
  { x: 56.0252, y: 71.7366 },
  { x: 41.5493, y: 92.3655 },
  { x: 70.7299, y: 92.2041 },
];

afterEach(() => vi.unstubAllGlobals());

describe("GhostFaceNet five point alignment", () => {
  it("maps eye centers, nose, and mouth corners into the 112px reference template", () => {
    const contexts = mockCanvas();
    const face: DetectedFace = {
      x: 0, y: 0, width: 300, height: 300, confidence: 0.99,
      landmarks: template.map(({ x, y }) => ({ x: x * 2 + 20, y: y * 2 + 30 })),
    };

    const aligned = alignFaceImage({ width: 640, height: 480 } as ImageData, face);

    expect(aligned.width).toBe(112);
    expect(aligned.height).toBe(112);
    const transform = contexts[1].setTransform.mock.calls[0];
    expect(transform).toHaveLength(6);
    expect(transform![0]).toBeCloseTo(.5);
    expect(transform![1]).toBeCloseTo(0);
    expect(transform![2]).toBeCloseTo(0);
    expect(transform![3]).toBeCloseTo(.5);
    expect(transform![4]).toBeCloseTo(-10);
    expect(transform![5]).toBeCloseTo(-15);
  });

  it("uses MediaPipe Face Mesh eye, nose, and mouth-corner indices", () => {
    const contexts = mockCanvas();
    const dense = Array.from({ length: 468 }, () => ({ x: 0, y: 0 }));
    [33, 263, 1, 61, 291].forEach((index, point) => { dense[index] = { x: template[point].x * 2, y: template[point].y * 2 }; });
    const face: DetectedFace = { x: 0, y: 0, width: 220, height: 220, confidence: .99, landmarks: dense };

    alignFaceImage({ width: 640, height: 480 } as ImageData, face);

    const transform = contexts[1].setTransform.mock.calls[0];
    expect(transform![0]).toBeCloseTo(.5);
    expect(transform![1]).toBeCloseTo(0);
    expect(transform![2]).toBeCloseTo(0);
    expect(transform![3]).toBeCloseTo(.5);
    expect(transform![4]).toBeCloseTo(0);
    expect(transform![5]).toBeCloseTo(0);
  });

  it("rejects malformed landmark geometry", () => {
    mockCanvas();
    const face: DetectedFace = { x: 0, y: 0, width: 100, height: 100, confidence: .9, landmarks: [] };
    expect(() => alignFaceImage({ width: 100, height: 100 } as ImageData, face)).toThrow(/five valid/i);
  });
});
