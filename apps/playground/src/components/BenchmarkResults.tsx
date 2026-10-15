import type { BenchmarkResult, TimingStats } from "../benchmark";
import { buildTimingPoints } from "../report";

type BenchmarkResultsProps = {
  result: BenchmarkResult;
};

export function BenchmarkResults({ result }: BenchmarkResultsProps) {
  const { metadata, stats, runs } = result;
  const width = 640;
  const height = 188;
  const points = buildTimingPoints(runs.map((run) => run.totalMs), width, height);
  const line = points.map(({ x, y }) => `${x},${y}`).join(" ");
  const area = `0,${height - 6} ${line} ${width},${height - 6}`;
  const stages: [string, TimingStats][] = [
    ["Preprocessing", stats.preprocessing],
    ["Inference", stats.inference],
    ["Decoding", stats.decoding],
  ];
  const averageFps = 1000 / stats.total.average;

  return (
    <div className="panel report-panel">
      <header className="report-header">
        <div>
          <p className="eyebrow">BENCHMARK</p>
          <h2 id="report-heading">{metadata.modelName} benchmark</h2>
          <p className="report-subtitle">
            {metadata.backend.toUpperCase()} backend <span>·</span> {metadata.threadCount} thread(s) <span>·</span> {metadata.inputSize.width} × {metadata.inputSize.height} input
          </p>
        </div>
        <div className="report-header-meta">
          <span className="report-success"><span className="online-dot" /> COMPLETE</span>
          <span>{metadata.measuredRuns} measured runs</span>
        </div>
      </header>

      <div className="report-stats">
        <article className="report-stat report-stat-primary">
          <span>AVERAGE LATENCY</span>
          <strong>{stats.total.average.toFixed(2)}<i>ms</i></strong>
          <small>per frame · end-to-end pipeline</small>
        </article>
        <article className="report-stat">
          <span>MEDIAN LATENCY</span>
          <strong>{stats.total.median.toFixed(2)}<i>ms</i></strong>
          <small>50th percentile</small>
        </article>
        <article className="report-stat">
          <span>P95 LATENCY</span>
          <strong>{stats.total.p95.toFixed(2)}<i>ms</i></strong>
          <small>95th percentile</small>
        </article>
        <article className="report-stat">
          <span>ESTIMATED FPS</span>
          <strong>{averageFps.toFixed(1)}<i>FPS</i></strong>
          <small>based on average latency</small>
        </article>
      </div>

      <div className="report-chart-layout">
        <section className="report-chart-card" aria-labelledby="latency-chart-title">
          <div className="chart-heading">
            <div><h3 id="latency-chart-title">Latency by run</h3><p>End-to-end time per measured frame</p></div>
            <span className="chart-legend"><i /> Total latency <b>·</b> ms</span>
          </div>
          <svg className="timing-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Total latency across ${runs.length} benchmark runs`} preserveAspectRatio="none">
            <defs>
              <linearGradient id="latency-fill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="var(--chart-line)" stopOpacity=".2" />
                <stop offset="100%" stopColor="var(--chart-line)" stopOpacity="0" />
              </linearGradient>
            </defs>
            {[8, height / 2, height - 6].map((y) => <line key={y} x1="0" x2={width} y1={y} y2={y} className="chart-gridline" />)}
            {points.length > 1 && <polygon points={area} fill="url(#latency-fill)" />}
            {points.length > 1 && <polyline points={line} className="chart-line" />}
            {points.map((point, index) => (
              (index % Math.max(1, Math.ceil(runs.length / 7)) === 0 || index === runs.length - 1) && (
                <circle key={index} cx={point.x} cy={point.y} r="3.5" className="chart-point">
                  <title>{`Run ${index + 1}: ${runs[index].totalMs.toFixed(2)} ms`}</title>
                </circle>
              )
            ))}
          </svg>
          <div className="chart-axis"><span>Run 01</span><span>Run {Math.ceil(runs.length / 2)}</span><span>Run {runs.length}</span></div>
        </section>

        <section className="report-breakdown" aria-labelledby="breakdown-title">
          <div className="breakdown-heading"><div><h3 id="breakdown-title">Stage breakdown</h3><p>Average time per frame</p></div><span>AVG · MS</span></div>
          <div className="stage-list">
            {stages.map(([label, timing], index) => (
              <div className="stage-row" key={label}>
                <div className="stage-label"><span className={`stage-dot stage-dot-${index}`} />{label}<strong>{timing.average.toFixed(2)}<i>ms</i></strong></div>
                <div className="stage-track"><span className={`stage-bar stage-bar-${index}`} style={{ width: `${Math.max(2, (timing.average / stats.total.average) * 100)}%` }} /></div>
              </div>
            ))}
          </div>
          <div className="breakdown-footer"><span>Faces in frozen frame</span><strong>{result.faceCount}</strong></div>
        </section>
      </div>

      <details className="run-details">
        <summary><span>Run details</span><small>{runs.length} measured frames · {metadata.warmupRuns} warmup runs excluded</small><span className="summary-chevron" aria-hidden="true">⌄</span></summary>
        <div className="table-scroll">
          <table className="timing-table">
            <thead><tr><th>RUN</th><th>PREPROCESSING</th><th>INFERENCE</th><th>DECODING</th><th>TOTAL</th></tr></thead>
            <tbody>
              {runs.map((run, index) => (
                <tr key={index}>
                  <td>{String(index + 1).padStart(2, "0")}</td>
                  <td>{run.preprocessingMs.toFixed(2)} ms</td>
                  <td>{run.inferenceMs.toFixed(2)} ms</td>
                  <td>{run.decodingMs.toFixed(2)} ms</td>
                  <td><strong>{run.totalMs.toFixed(2)} ms</strong></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <footer className="report-footer"><span>Units: ms</span><span>{metadata.warmupRuns} warmups · {metadata.measuredRuns} measured · {result.faceCount} faces in frozen frame</span></footer>
    </div>
  );
}
