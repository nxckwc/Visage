import type { BenchmarkResult, TimingStats } from "../benchmark";

type BenchmarkResultsProps = {
  result: BenchmarkResult;
};

export function BenchmarkResults({ result }: BenchmarkResultsProps) {
  const { metadata, stats, runs } = result;
  const rows: [string, TimingStats][] = [
    ["Preprocessing", stats.preprocessing],
    ["Inference", stats.inference],
    ["Decoding", stats.decoding],
    ["Total", stats.total],
  ];

  return (
    <section style={{ marginTop: 24 }}>
      <h3>{metadata.modelName} Benchmark Results</h3>
      <p>
        {metadata.backend} · {metadata.threadCount} thread(s) ·{" "}
        {metadata.inputSize.width}×{metadata.inputSize.height} ·{" "}
        {metadata.warmupRuns} warmups · {metadata.measuredRuns} measured runs
      </p>
      <p>Faces detected: {result.faceCount}</p>

      <table style={{ width: "100%", textAlign: "right" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left" }}>Stage</th>
            <th>Avg</th>
            <th>Median</th>
            <th>Fastest</th>
            <th>P95</th>
            <th>Slowest</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, timing]) => (
            <tr key={label}>
              <td style={{ textAlign: "left" }}>{label}</td>
              <td>{timing.average.toFixed(1)}</td>
              <td>{timing.median.toFixed(1)}</td>
              <td>{timing.min.toFixed(1)}</td>
              <td>{timing.p95.toFixed(1)}</td>
              <td>{timing.max.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <details>
        <summary>Per-run timings</summary>
        <table style={{ width: "100%", textAlign: "right" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>Run</th>
              <th>Preprocessing</th>
              <th>Inference</th>
              <th>Decoding</th>
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run, index) => (
              <tr key={index}>
                <td style={{ textAlign: "left" }}>{index + 1}</td>
                <td>{run.preprocessingMs.toFixed(2)}</td>
                <td>{run.inferenceMs.toFixed(2)}</td>
                <td>{run.decodingMs.toFixed(2)}</td>
                <td>{run.totalMs.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
      <p>All timings are in milliseconds (ms).</p>
    </section>
  );
}
