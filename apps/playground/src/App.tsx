import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

import { runBenchmark, type BenchmarkResult } from "./benchmark";
import { BenchmarkResults } from "./components/BenchmarkResults";
import { YuNetDetector } from "./detectors/yunet";
import { SCRFDDetector } from "./detectors/scrfd";
import type { DetectedFace, DetectorTimings, FaceDetector } from "./detectors/types";

const THREAD_COUNT = 1;
const WARMUP_RUNS = 5;
const MEASURED_RUNS = 30;

const DETECTOR_NAMES = {
  yunet: "YuNet",
  scrfd: "SCRFD-500M",
} as const;

type ModelId = keyof typeof DETECTOR_NAMES;

type PoseIndicators = {
  rollDegrees: number;
  yawIndex: number;
  pitchRatio: number;
};

type FrameResult = {
  faceCount: number;
  confidence: number;
  timings: DetectorTimings;
  pose: PoseIndicators | null;
};

function estimatePose(face: DetectedFace): PoseIndicators | null {
  if (face.landmarks.length < 5) return null;

  const [eyeA, eyeB, nose, mouthA, mouthB] = face.landmarks;
  const eyeDx = eyeB.x - eyeA.x;
  const eyeDy = eyeB.y - eyeA.y;
  const eyeDistance = Math.hypot(eyeDx, eyeDy);
  if (eyeDistance < 1) return null;

  const eyeCenter = {
    x: (eyeA.x + eyeB.x) / 2,
    y: (eyeA.y + eyeB.y) / 2,
  };
  const mouthCenter = {
    x: (mouthA.x + mouthB.x) / 2,
    y: (mouthA.y + mouthB.y) / 2,
  };

  // Rotate into an eye-aligned coordinate system. This reduces
  // interference from in-plane head roll.
  const ux = eyeDx / eyeDistance;
  const uy = eyeDy / eyeDistance;
  const vx = -uy;
  const vy = ux;

  const noseX = (nose.x - eyeCenter.x) * ux + (nose.y - eyeCenter.y) * uy;
  const eyeToNoseY = (nose.x - eyeCenter.x) * vx + (nose.y - eyeCenter.y) * vy;
  const noseToMouthY =
    (mouthCenter.x - nose.x) * vx + (mouthCenter.y - nose.y) * vy;

  return {
    rollDegrees: (Math.atan2(eyeDy, eyeDx) * 180) / Math.PI,
    yawIndex: (2 * noseX) / eyeDistance,
    pitchRatio:
      Math.abs(noseToMouthY) > 0.001
        ? eyeToNoseY / noseToMouthY
        : Number.NaN,
  };
}

function drawFaces(
  context: CanvasRenderingContext2D,
  faces: DetectedFace[]
): void {
  context.lineWidth = 3;
  context.font = "16px monospace";

  for (const face of faces) {
    context.strokeStyle = "#00ff88";
    context.strokeRect(face.x, face.y, face.width, face.height);

    context.fillStyle = "#00ff88";
    context.fillText(
      `${(face.confidence * 100).toFixed(1)}%`,
      face.x,
      Math.max(16, face.y - 6)
    );

    for (const point of face.landmarks) {
      context.beginPath();
      context.arc(point.x, point.y, 4, 0, Math.PI * 2);
      context.fillStyle = "#ff4444";
      context.fill();
    }
  }
}

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
    pose: estimatePose(best),
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
  const [status, setStatus] = useState("Ready");
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

      setStatus("Tracking faces live...");
      while (trackingRef.current && generation === generationRef.current) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame = context.getImageData(0, 0, canvas.width, canvas.height);
        const { faces, timings } = await detector.detect(frame);

        if (!trackingRef.current || generation !== generationRef.current) return;

        const overlayContext = overlayRef.current?.getContext("2d");
        if (overlayContext) {
          overlayContext.clearRect(0, 0, canvas.width, canvas.height);
          drawFaces(overlayContext, faces);
        }

        setLiveResult(summarizeDetection(faces, timings));
        setStatus(faces.length ? `Tracking ${faces.length} face(s)` : "Searching for faces...");

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
    setStatus("Starting camera...");

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
      setStatus("Wait for the camera to start");
      return;
    }

    // Snapshot the frame *at the instant Bulk Test is clicked*.
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context || !canvas.width || !canvas.height) {
      setStatus("Could not freeze camera frame");
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
    setStatus("Frame frozen. Waiting for active inference...");

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
        setStatus("Benchmark complete — camera stopped");
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

  const livePose = liveResult?.pose;
  const displayed = frozenResult ?? liveResult;

  return (
    <main style={{ padding: 32, maxWidth: 900, margin: "auto" }}>
      <h1>Visage Playground</h1>
      <h2>Live face tracking and pose indicators</h2>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}>
        <label htmlFor="model">Model</label>
        <select
          id="model"
          value={selectedModel}
          disabled={isTracking || busy || Boolean(captureTaskRef.current)}
          onChange={(event) => {
            setSelectedModel(event.target.value as ModelId);
            setLiveResult(null);
            setFrozenResult(null);
            setBenchmarkResults(null);
            setStatus("Ready");
          }}
        >
          {Object.entries(DETECTOR_NAMES).map(([id, name]) => (
            <option key={id} value={id}>{name}</option>
          ))}
        </select>

        <button onClick={startCamera} disabled={isTracking || busy}>
          Start Camera
        </button>
        <button onClick={stopTracking} disabled={!isTracking || busy}>
          Stop
        </button>
        <button onClick={() => void bulkTest()} disabled={!isTracking || busy}>
          {busy ? `Benchmarking ${benchmarkProgress}/${MEASURED_RUNS}` : "Bulk Test (30×)"}
        </button>
      </div>

      <p>Status: {status}</p>

      <div style={{ position: "relative", width: "100%", marginTop: 12 }}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{ display: isTracking ? "block" : "none", width: "100%" }}
        />
        <canvas
          ref={overlayRef}
          style={{
            display: isTracking ? "block" : "none",
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
          }}
        />
        <canvas
          ref={frozenCanvasRef}
          style={{ display: !isTracking && frozenResult ? "block" : "none", width: "100%" }}
        />
      </div>

      {isTracking && (
        <section>
          <h3>Live pose (landmark-based estimates)</h3>
          <p>Roll: {livePose ? `${livePose.rollDegrees.toFixed(1)}°` : "—"}</p>
          <p>Yaw asymmetry: {livePose ? livePose.yawIndex.toFixed(3) : "—"}</p>
          <p>Pitch ratio: {livePose && Number.isFinite(livePose.pitchRatio)
            ? livePose.pitchRatio.toFixed(3) : "—"}</p>
          <p>Faces: {liveResult?.faceCount ?? 0}</p>
          <p style={{ fontSize: 13 }}>
            Yaw and pitch are <strong>uncalibrated geometric indicators</strong>, not 3D angles.
            Their sign depends on landmark order and camera orientation.
          </p>
        </section>
      )}

      {!isTracking && frozenResult && (
        <section>
          <h3>Frozen frame</h3>
          <p>Faces: {frozenResult.faceCount}</p>
          <p>Detection confidence: {(frozenResult.confidence * 100).toFixed(1)}%</p>
          <p>Roll: {frozenResult.pose?.rollDegrees.toFixed(1) ?? "—"}°</p>
          <p>Yaw asymmetry: {frozenResult.pose?.yawIndex.toFixed(3) ?? "—"}</p>
          <p>Pitch ratio: {frozenResult.pose && Number.isFinite(frozenResult.pose.pitchRatio)
            ? frozenResult.pose.pitchRatio.toFixed(3) : "—"}</p>
          <p>Preprocess: {frozenResult.timings.preprocessingMs.toFixed(1)} ms</p>
          <p>Inference: {frozenResult.timings.inferenceMs.toFixed(1)} ms</p>
          <p>Decode: {frozenResult.timings.decodingMs.toFixed(1)} ms</p>
        </section>
      )}

      {benchmarkResults && <BenchmarkResults result={benchmarkResults} />}
      {!isTracking && busy && <p>Benchmark progress: {benchmarkProgress}/{MEASURED_RUNS}</p>}
      {displayed && !isTracking && !frozenResult && <p>Camera stopped.</p>}
    </main>
  );
}
