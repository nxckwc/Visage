import { describe, expect, it, vi } from "vitest";
import type { DetectedFace, DetectorTimings, FaceDetector } from "../src/detectors/types";
import { BlazeFaceLandmarkPipeline } from "../src/detectors/landmarkPipeline";

const frame = {
  width: 100,
  height: 100,
  data: new Uint8ClampedArray(100 * 100 * 4),
} as ImageData;

const alignedFace: DetectedFace = {
  x: 20,
  y: 30,
  width: 20,
  height: 40,
  confidence: 0.9,
  landmarks: [
    { x: 30, y: 40 },
    { x: 40, y: 40 },
    { x: 35, y: 48 },
    { x: 32, y: 55 },
    { x: 38, y: 55 },
    { x: 45, y: 42 },
  ],
};

const timing = (preprocessingMs: number, inferenceMs: number, decodingMs: number): DetectorTimings => ({
  preprocessingMs,
  inferenceMs,
  decodingMs,
});

function detector(
  id: string,
  name: string,
  result: { faces: DetectedFace[]; timings: DetectorTimings }
): FaceDetector {
  return {
    id,
    name,
    inputSize: { width: 128, height: 128 },
    load: vi.fn(async () => undefined),
    detect: vi.fn(async () => result),
    dispose: vi.fn(async () => undefined),
  };
}

function installCanvasMock() {
  const context = {
    fillStyle: "",
    fillRect: vi.fn(),
    putImageData: vi.fn(),
    getImageData: vi.fn((_x: number, _y: number, width: number, height: number) => ({
      width,
      height,
      data: new Uint8ClampedArray(width * height * 4),
    })),
    save: vi.fn(),
    restore: vi.fn(),
    setTransform: vi.fn(),
    translate: vi.fn(),
    rotate: vi.fn(),
    scale: vi.fn(),
    drawImage: vi.fn(),
  };
  vi.stubGlobal("document", {
    createElement: vi.fn(() => ({
      width: 0,
      height: 0,
      getContext: vi.fn(() => context),
    })),
  });
}

describe("BlazeFace to MediaPipe landmark pipeline", () => {
  it("aligns detected face and maps dense landmarks back to source coordinates", async () => {
    installCanvasMock();
    const aligner = detector("blazeface", "BlazeFace", {
      faces: [alignedFace],
      timings: timing(10, 20, 30),
    });
    const denseFace: DetectedFace = {
      ...alignedFace,
      x: 100,
      y: 100,
      width: 30,
      height: 30,
      landmarks: Array.from({ length: 468 }, () => ({ x: 128, y: 128 })),
    };
    const landmarker = detector("mediapipe", "MediaPipe", {
      faces: [denseFace],
      timings: timing(1, 2, 3),
    });
    const pipeline = new BlazeFaceLandmarkPipeline(aligner, landmarker);

    await pipeline.load();
    const result = await pipeline.detect(frame);

    expect(aligner.load).toHaveBeenCalledOnce();
    expect(landmarker.load).toHaveBeenCalledOnce();
    expect(landmarker.detect).toHaveBeenCalledOnce();
    expect(landmarker.detect.mock.calls[0][0].width).toBe(256);
    expect(landmarker.detect.mock.calls[0][0].height).toBe(256);
    expect(result.faces).toHaveLength(1);
    expect(result.faces[0].confidence).toBe(alignedFace.confidence);
    expect(result.faces[0].landmarks).toHaveLength(468);
    expect(result.faces[0].landmarks[0]).toEqual({ x: 30, y: 50 });
    expect(result.timings.inferenceMs).toBe(22);
    expect(result.timings.decodingMs).toBeGreaterThanOrEqual(33);

    await pipeline.dispose();
  });

  it("waits for active refinement before disposing child detectors", async () => {
    installCanvasMock();
    const aligner = detector("blazeface", "BlazeFace", {
      faces: [alignedFace],
      timings: timing(1, 2, 3),
    });
    const landmarker = detector("mediapipe", "MediaPipe", {
      faces: [],
      timings: timing(0, 0, 0),
    });
    let finishRefinement!: (value: { faces: DetectedFace[]; timings: DetectorTimings }) => void;
    landmarker.detect = vi.fn(() => new Promise((resolve) => {
      finishRefinement = resolve;
    }));
    const pipeline = new BlazeFaceLandmarkPipeline(aligner, landmarker);

    await pipeline.load();
    const running = pipeline.detect(frame);
    await vi.waitFor(() => expect(landmarker.detect).toHaveBeenCalledOnce());
    const disposal = pipeline.dispose();

    expect(aligner.dispose).not.toHaveBeenCalled();
    expect(landmarker.dispose).not.toHaveBeenCalled();
    finishRefinement({ faces: [], timings: timing(0, 0, 0) });
    await running;
    await disposal;

    expect(aligner.dispose).toHaveBeenCalledOnce();
    expect(landmarker.dispose).toHaveBeenCalledOnce();
  });

  it("filters detections below 60% and drops faces MediaPipe cannot refine", async () => {
    installCanvasMock();
    const lowConfidenceFace = { ...alignedFace, confidence: 0.599 };
    const thresholdFace = { ...alignedFace, confidence: 0.6 };
    const aligner = detector("blazeface", "BlazeFace", {
      faces: [lowConfidenceFace, thresholdFace],
      timings: timing(1, 2, 3),
    });
    const landmarker = detector("mediapipe", "MediaPipe", {
      faces: [],
      timings: timing(4, 5, 6),
    });
    const pipeline = new BlazeFaceLandmarkPipeline(aligner, landmarker);

    await pipeline.load();
    const result = await pipeline.detect(frame);

    expect(landmarker.detect).toHaveBeenCalledOnce();
    expect(result.faces).toEqual([]);
    await pipeline.dispose();
  });

  it("does not run the landmarker when alignment finds no face", async () => {
    installCanvasMock();
    const aligner = detector("blazeface", "BlazeFace", {
      faces: [],
      timings: timing(1, 2, 3),
    });
    const landmarker = detector("mediapipe", "MediaPipe", {
      faces: [],
      timings: timing(0, 0, 0),
    });
    const pipeline = new BlazeFaceLandmarkPipeline(aligner, landmarker);

    await pipeline.load();
    const result = await pipeline.detect(frame);

    expect(landmarker.detect).not.toHaveBeenCalled();
    expect(result.faces).toEqual([]);
    expect(result.timings).toEqual(timing(1, 2, 3));
    await pipeline.dispose();
  });
});
