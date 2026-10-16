import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

import { runBenchmark, type BenchmarkResult } from "./benchmark";
import { BenchmarkResults } from "./components/BenchmarkResults";
import { FaceRegister } from "./components/FaceRegister";
import { FaceRecognition } from "./components/FaceRecognition";
import { YuNetDetector } from "./detectors/yunet";
import { SCRFDDetector } from "./detectors/scrfd";
import { BlazeFaceLandmarkPipeline } from "./detectors/landmarkPipeline";
import {
  BlazeFaceShortRangeDetector,
  MediaPipeLandmarkDetector,
} from "./detectors/blazeface";
import type { DetectedFace, DetectorTimings, FaceDetector } from "./detectors/types";
import { drawFaces, smoothFaces } from "./detectionOverlay";

const THREAD_COUNT = 1;
const WARMUP_RUNS = 5;
const MEASURED_RUNS = 30;

const DETECTOR_NAMES = {
  yunet: "YuNet",
  scrfd: "SCRFD-500M",
  blazeface: "MediaPipe BlazeFace Short Range (6 landmarks)",
  mediapipe: "BlazeFace → MediaPipe 468 landmarks",
} as const;

type ModelId = keyof typeof DETECTOR_NAMES;
type TabId = "detection" | "registration" | "recognition";

type FrameResult = {
  faceCount: number;
  confidence: number;
  timings: DetectorTimings;
};

function summarizeDetection(
  faces: DetectedFace[],
  timings: DetectorTimings
): FrameResult | null {
  if (faces.length === 0) return null;
  const best = faces.reduce((a, b) =>
    b.confidence > a.confidence ? b : a
  );
  return {
    faceCount: faces.length,
    confidence: best.confidence,
    timings,
  };
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const frozenCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectorsRef = useRef<Record<ModelId, FaceDetector> | null>(null);

  const mountedRef = useRef(false);
  const trackingRef = useRef(false);
  const captureTaskRef = useRef<Promise<void> | null>(null);
  const generationRef = useRef(0);
  const benchmarkControllerRef = useRef<AbortController | null>(null);
  const benchmarkActiveRef = useRef(false);

  const [selectedModel, setSelectedModel] = useState<ModelId>("yunet");
  const [activeTab, setActiveTab] = useState<TabId>("detection");
  const [status, setStatus] = useState("Idle");
  const [isTracking, setIsTracking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [liveResult, setLiveResult] = useState<FrameResult | null>(null);
  const [frozenResult, setFrozenResult] = useState<FrameResult | null>(null);
  const [benchmarkProgress, setBenchmarkProgress] = useState(0);
  const [benchmarkResults, setBenchmarkResults] =
    useState<BenchmarkResult | null>(null);

  function closeCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }

  function stopTracking() {
    trackingRef.current = false;
    generationRef.current++;
    closeCamera();
    if (mountedRef.current) {
      setIsTracking(false);
      setStatus("Stopped");
    }
  }

  useEffect(() => {
    const detectors: Record<ModelId, FaceDetector> = {
      yunet: new YuNetDetector(),
      scrfd: new SCRFDDetector(),
      blazeface: new BlazeFaceShortRangeDetector(),
      mediapipe: new BlazeFaceLandmarkPipeline(
        new BlazeFaceShortRangeDetector(),
        new MediaPipeLandmarkDetector()
      ),
    };
    detectorsRef.current = detectors;
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      trackingRef.current = false;
      generationRef.current++;
      benchmarkControllerRef.current?.abort();
      closeCamera();
      detectorsRef.current = null;

      // Both adapters must wait for an active detect() to settle
      // before releasing their ONNX inference sessions.
      void Promise.all(Object.values(detectors).map((d) => d.dispose()))
        .catch(console.error);
    };
  }, []);

  async function trackCamera(detector: FaceDetector, generation: number) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
        audio: false,
      });
      if (!trackingRef.current || generation !== generationRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;

      const video = videoRef.current;
      if (!video) throw new Error("Video element unavailable");
      video.srcObject = stream;
      await video.play();
      if (!trackingRef.current || generation !== generationRef.current) return;

      ort.env.wasm.numThreads = THREAD_COUNT;
      setStatus(`Loading ${detector.name}...`);
      await detector.load();
      if (!trackingRef.current || generation !== generationRef.current) return;

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      if (!canvas.width || !canvas.height) {
        throw new Error("Camera has no video dimensions");
      }
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) throw new Error("Cannot create camera canvas");

      const overlay = overlayRef.current;
      if (overlay) {
        overlay.width = canvas.width;
        overlay.height = canvas.height;
      }

      let previousFaces: DetectedFace[] = [];
      setStatus("Detection active");
      while (trackingRef.current && generation === generationRef.current) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame = context.getImageData(0, 0, canvas.width, canvas.height);
        const { faces, timings } = await detector.detect(frame);

        if (!trackingRef.current || generation !== generationRef.current) return;

        const displayFaces = smoothFaces(previousFaces, faces);
        previousFaces = displayFaces;
        const overlayContext = overlayRef.current?.getContext("2d");
        if (overlayContext) {
          overlayContext.clearRect(0, 0, canvas.width, canvas.height);
          drawFaces(overlayContext, displayFaces);
        }

        setLiveResult(summarizeDetection(faces, timings));
        setStatus(faces.length ? `Detected ${faces.length} face(s)` : "No detections");

        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      }
    } catch (error) {
      if (mountedRef.current && generation === generationRef.current) {
        setStatus(`Camera/detection error: ${String(error)}`);
      }
    } finally {
      if (generation === generationRef.current) {
        trackingRef.current = false;
        closeCamera();
        if (mountedRef.current) setIsTracking(false);
      }
    }
  }

  function startCamera() {
    if (
      trackingRef.current ||
      captureTaskRef.current ||
      benchmarkActiveRef.current
    ) return;

    const detector = detectorsRef.current?.[selectedModel];
    if (!detector) return;

    const generation = ++generationRef.current;
    trackingRef.current = true;
    setIsTracking(true);
    setLiveResult(null);
    setFrozenResult(null);
    setBenchmarkResults(null);
    setBenchmarkProgress(0);
    setStatus("Starting camera");

    const task = trackCamera(detector, generation);
    captureTaskRef.current = task;
    void task.finally(() => {
      if (captureTaskRef.current === task) captureTaskRef.current = null;
    });
  }

  async function bulkTest() {
    if (!trackingRef.current || benchmarkActiveRef.current) return;
    const detector = detectorsRef.current?.[selectedModel];
    const video = videoRef.current;
    if (!detector || !video || video.readyState < 2) {
      setStatus("Camera is not ready");
      return;
    }

    // Snapshot the frame *at the instant Bulk Test is clicked*.
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context || !canvas.width || !canvas.height) {
      setStatus("Frame capture failed");
      return;
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frozenFrame = context.getImageData(0, 0, canvas.width, canvas.height);

    const controller = new AbortController();
    benchmarkControllerRef.current = controller;
    benchmarkActiveRef.current = true;
    setBusy(true);
    setBenchmarkResults(null);
    setBenchmarkProgress(0);

    // Stop camera immediately, then wait for any in-flight detect()
    // before starting the benchmark on the same detector session.
    stopTracking();
    setStatus("Frame captured; waiting for inference");

    try {
      await captureTaskRef.current;
      controller.signal.throwIfAborted();

      // Annotate frozen frame once. This inference is NOT counted
      // in the measured benchmark runs.
      const { faces, timings } = await detector.detect(frozenFrame);
      controller.signal.throwIfAborted();
      const outputCanvas = frozenCanvasRef.current;
      if (outputCanvas) {
        outputCanvas.width = frozenFrame.width;
        outputCanvas.height = frozenFrame.height;
        const outputContext = outputCanvas.getContext("2d");
        if (outputContext) {
          outputContext.putImageData(frozenFrame, 0, 0);
          drawFaces(outputContext, faces);
        }
      }
      if (mountedRef.current) setFrozenResult(summarizeDetection(faces, timings));

      const results = await runBenchmark(detector, frozenFrame, {
        warmupRuns: WARMUP_RUNS,
        measuredRuns: MEASURED_RUNS,
        backend: "wasm",
        threadCount: THREAD_COUNT,
        signal: controller.signal,
        onProgress: (completed, total) => {
          if (mountedRef.current && !controller.signal.aborted) {
            setBenchmarkProgress(completed);
            setStatus(`Benchmarking ${completed}/${total}...`);
          }
        },
      });

      if (mountedRef.current && !controller.signal.aborted) {
        setBenchmarkResults(results);
        setStatus("Benchmark complete");
      }
    } catch (error) {
      if (mountedRef.current && !controller.signal.aborted) {
        console.error(error);
        setStatus(`Benchmark failed: ${String(error)}`);
      }
    } finally {
      if (benchmarkControllerRef.current === controller) {
        benchmarkControllerRef.current = null;
      }
      benchmarkActiveRef.current = false;
      if (mountedRef.current) setBusy(false);
    }
  }

  const currentResult = isTracking ? liveResult : frozenResult ?? liveResult;

  return (
    <div className="app-shell">
      <aside className="app-sidebar" aria-label="Main navigation">
        <a className="brand-mark" href="#playground" aria-label="Visage home"><span className="brand-icon">V</span><span>visage</span></a>
        <span className="sidebar-label">PLAYGROUND</span>
        <nav className="sidebar-nav">
          <button className={`nav-item ${activeTab === "detection" ? "is-active" : ""}`} onClick={() => { if (isTracking) stopTracking(); setActiveTab("detection"); }}><span className="nav-icon">◎</span><span>Face detection</span></button>
          <button className={`nav-item ${activeTab === "registration" ? "is-active" : ""}`} onClick={() => { if (isTracking) stopTracking(); setActiveTab("registration"); }}><span className="nav-icon">▤</span><span>Face register</span><span className="nav-badge">NEW</span></button>
          <button className={`nav-item ${activeTab === "recognition" ? "is-active" : ""}`} onClick={() => { if (isTracking) stopTracking(); setActiveTab("recognition"); }}><span className="nav-icon">◎</span><span>Face recognition</span></button>
        </nav>
        <div className="sidebar-footer"><span className="sidebar-footer-mark">●</span><span>Local AI playground<small>Models run in your browser</small></span></div>
      </aside>
      <main className="playground">
      <header className="page-header">
        <div><span className="page-kicker">VISAGE / PLAYGROUND</span><h1 className="page-title">{activeTab === "detection" ? "Face detection" : activeTab === "registration" ? "Face register" : "Face recognition"}</h1></div>
        <span className={`session-status ${activeTab === "detection" && isTracking ? "is-live" : busy ? "is-busy" : ""}`}>
          <span className="status-dot" />
          {activeTab === "registration" ? "Ready" : busy ? "Benchmark running" : isTracking ? "Camera live" : "Idle"}
        </span>
      </header>

      {activeTab === "registration" ? <div className="page-content register-content" id="playground"><div className="view-intro"><div><p className="eyebrow">IDENTITY WORKSPACE</p><h2>Build your face register</h2><p>Capture a face, give it a name, and keep its recognition data on this device. Face crops are now aligned using eye landmarks; previously registered faces need to be added again.</p></div><span className="privacy-chip"><span className="online-dot" /> PRIVATE BY DESIGN</span></div><FaceRegister detectors={detectorsRef.current!} /></div> : activeTab === "recognition" ? <FaceRecognition detectors={detectorsRef.current!} /> : <div className="page-content" id="playground">
        <section className="camera-panel" aria-labelledby="camera-title">
          <div className="camera-toolbar">
            <div className="camera-heading">
              <h2 id="camera-title">Camera input</h2>
              <span className={`live-indicator ${isTracking ? "is-live" : ""}`}>
                <span className="status-dot" />{isTracking ? "LIVE" : busy ? "RUNNING" : "IDLE"}
              </span>
            </div>
            <div className="camera-controls">
              <label className="visually-hidden" htmlFor="model">Detection model</label>
              <select
                id="model"
                value={selectedModel}
                disabled={isTracking || busy}
                onChange={(event) => {
                  setSelectedModel(event.target.value as ModelId);
                  setLiveResult(null);
                  setFrozenResult(null);
                  setBenchmarkResults(null);
                  setStatus("Idle");
                }}
              >
                {Object.entries(DETECTOR_NAMES).map(([id, name]) => (
                  <option key={id} value={id}>{name}</option>
                ))}
              </select>
              {isTracking ? (
                <button className="button button-secondary" onClick={stopTracking} disabled={busy}>Stop</button>
              ) : (
                <button className="button button-primary" onClick={startCamera} disabled={busy}>
                  <span aria-hidden="true">▶</span> Start camera
                </button>
              )}
              <button className="button button-benchmark" onClick={() => void bulkTest()} disabled={!isTracking || busy}>
                {busy ? `Benchmarking ${benchmarkProgress}/${MEASURED_RUNS}` : "Run benchmark"}
              </button>
            </div>
          </div>

          <div className={`camera-stage ${isTracking || frozenResult ? "camera-active" : ""}`}>
            <video ref={videoRef} autoPlay muted playsInline className={isTracking ? "camera-media visible" : "camera-media"} />
            <canvas ref={overlayRef} className={isTracking ? "camera-overlay visible" : "camera-overlay"} />
            <canvas ref={frozenCanvasRef} className={!isTracking && frozenResult ? "camera-frozen visible" : "camera-frozen"} />
            {!isTracking && !frozenResult && (
              <div className="camera-empty">
                <strong>No frame</strong>
              </div>
            )}
            <div className="camera-stage-footer">
              <span><span className={`status-dot ${isTracking ? "is-live" : ""}`} />{status}</span>
              <span>Local processing · frames not stored</span>
            </div>
          </div>

          <div className="live-stats" aria-live="polite">
            <div><span>Faces detected</span><strong>{currentResult?.faceCount ?? "—"}</strong></div>
            <div><span>Best confidence</span><strong>{currentResult ? `${(currentResult.confidence * 100).toFixed(1)}%` : "—"}</strong></div>
            <div><span>Inference time</span><strong>{currentResult ? <>{currentResult.timings.inferenceMs.toFixed(1)}<small> ms</small></> : "—"}</strong></div>
            <div className="selected-model"><span>Model</span><strong>{DETECTOR_NAMES[selectedModel]} <small>· WASM</small></strong></div>
          </div>
        </section>

        <section className="report-section" id="report" aria-labelledby="report-heading">
          <div className="section-heading">
            <div><h2 id="report-heading">Benchmark</h2><p>30 measured runs · 5 warm-up runs excluded.</p></div>
          </div>
          {benchmarkResults ? (
            <BenchmarkResults result={benchmarkResults} />
          ) : (
            <div className="report-placeholder">
              {busy ? (
                <>
                  <div className="progress-track" role="progressbar" aria-label="Benchmark progress" aria-valuemin={0} aria-valuemax={MEASURED_RUNS} aria-valuenow={benchmarkProgress}>
                    <span style={{ width: `${(benchmarkProgress / MEASURED_RUNS) * 100}%` }} />
                  </div>
                  <span>Measuring frame · {benchmarkProgress} of {MEASURED_RUNS} runs</span>
                </>
              ) : (
                <span>Run benchmark to record latency and stage timings.</span>
              )}
            </div>
          )}
        </section>

      </div>
      }
      </main>
    </div>
  );
}
