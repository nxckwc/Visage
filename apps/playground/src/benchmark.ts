import type { FaceDetector, DetectorTimings } from "./detectors/types";

export type TimingStats = {
  average: number;
  median: number;
  min: number;
  max: number;
  p95: number;
};

export type BenchmarkRun = DetectorTimings & { totalMs: number };

export type BenchmarkOptions = {
  warmupRuns?: number;
  measuredRuns?: number;
  signal?: AbortSignal;
  onProgress?: (completed: number, total: number) => void;
  backend: string;
  threadCount: number;
};

export type BenchmarkResult = {
  metadata: {
    modelId: string;
    modelName: string;
    inputSize: { width: number; height: number };
    backend: string;
    threadCount: number;
    warmupRuns: number;
    measuredRuns: number;
  };
  runs: BenchmarkRun[];
  stats: {
    preprocessing: TimingStats;
    inference: TimingStats;
    decoding: TimingStats;
    total: TimingStats;
  };
  faceCount: number;
};

export function calculateStats(values: number[]): TimingStats {
  if (values.length === 0) throw new Error("Cannot calculate stats without runs");
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;

  return {
    average: values.reduce((sum, value) => sum + value, 0) / n,
    median: n % 2 === 0
      ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
      : sorted[Math.floor(n / 2)],
    min: sorted[0],
    max: sorted[n - 1],
    p95: sorted[Math.ceil(n * 0.95) - 1],
  };
}

export async function runBenchmark(
  detector: FaceDetector,
  frame: ImageData,
  options: BenchmarkOptions
): Promise<BenchmarkResult> {
  const warmupRuns = options.warmupRuns ?? 5;
  const measuredRuns = options.measuredRuns ?? 30;
  if (!Number.isInteger(warmupRuns) || warmupRuns < 0) {
    throw new RangeError("warmupRuns must be a non-negative integer");
  }
  if (!Number.isInteger(measuredRuns) || measuredRuns < 1) {
    throw new RangeError("measuredRuns must be a positive integer");
  }
  if (!Number.isInteger(options.threadCount) || options.threadCount < 1) {
    throw new RangeError("threadCount must be a positive integer");
  }

  const runs: BenchmarkRun[] = [];
  let faceCount = 0;
  const totalRuns = warmupRuns + measuredRuns;

  for (let i = 0; i < totalRuns; i++) {
    options.signal?.throwIfAborted();
    const { faces, timings } = await detector.detect(frame);
    options.signal?.throwIfAborted();

    if (i >= warmupRuns) {
      runs.push({
        ...timings,
        totalMs: timings.preprocessingMs + timings.inferenceMs + timings.decodingMs,
      });
      faceCount = faces.length;
      options.onProgress?.(i - warmupRuns + 1, measuredRuns);
      options.signal?.throwIfAborted();
    }
  }

  return {
    metadata: {
      modelId: detector.id,
      modelName: detector.name,
      inputSize: detector.inputSize,
      backend: options.backend,
      threadCount: options.threadCount,
      warmupRuns,
      measuredRuns,
    },
    runs,
    stats: {
      preprocessing: calculateStats(runs.map((run) => run.preprocessingMs)),
      inference: calculateStats(runs.map((run) => run.inferenceMs)),
      decoding: calculateStats(runs.map((run) => run.decodingMs)),
      total: calculateStats(runs.map((run) => run.totalMs)),
    },
    faceCount,
  };
}
