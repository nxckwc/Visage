import { describe, expect, it, vi } from "vitest";
import type { FaceDetector, DetectorTimings } from "../src/detectors/types";
import { runBenchmark } from "../src/benchmark";

const frame = {
  width: 2,
  height: 2,
  data: new Uint8ClampedArray(16),
} as ImageData;

function detectorWithTimings(samples: DetectorTimings[]): FaceDetector {
  let index = 0;
  return {
    id: "yunet",
    name: "YuNet",
    inputSize: { width: 640, height: 640 },
    load: vi.fn(async () => undefined),
    detect: vi.fn(async () => ({
      faces: [],
      timings: samples[index++ % samples.length],
    })),
    dispose: vi.fn(async () => undefined),
  };
}

describe("runBenchmark", () => {
  it("retains each measured timing and benchmark metadata", async () => {
    const detector = detectorWithTimings([
      { preprocessingMs: 1, inferenceMs: 10, decodingMs: 0.2 },
      { preprocessingMs: 2, inferenceMs: 20, decodingMs: 0.4 },
      { preprocessingMs: 3, inferenceMs: 30, decodingMs: 0.6 },
    ]);
    const onProgress = vi.fn();

    const result = await runBenchmark(detector, frame, {
      warmupRuns: 1,
      measuredRuns: 3,
      backend: "wasm",
      threadCount: 1,
      onProgress,
    });

    expect(result.metadata).toEqual({
      modelId: "yunet",
      modelName: "YuNet",
      inputSize: { width: 640, height: 640 },
      backend: "wasm",
      threadCount: 1,
      warmupRuns: 1,
      measuredRuns: 3,
    });
    expect(result.runs.map((run) => run.inferenceMs)).toEqual([20, 30, 10]);
    expect(result.stats.inference.median).toBe(20);
    expect(onProgress).toHaveBeenLastCalledWith(3, 3);
  });

  it("stops after an in-flight detection settles when aborted", async () => {
    const controller = new AbortController();
    const detector = detectorWithTimings([
      { preprocessingMs: 1, inferenceMs: 2, decodingMs: 3 },
    ]);
    vi.mocked(detector.detect).mockImplementation(async () => {
      controller.abort();
      return {
        faces: [],
        timings: { preprocessingMs: 1, inferenceMs: 2, decodingMs: 3 },
      };
    });

    await expect(
      runBenchmark(detector, frame, {
        warmupRuns: 0,
        measuredRuns: 5,
        backend: "wasm",
        threadCount: 1,
        signal: controller.signal,
      })
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(detector.detect).toHaveBeenCalledOnce();
  });
});
