import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import type { FaceDetector, DetectedFace } from "../detectors/types";
import { drawFaces, smoothFaces } from "../detectionOverlay";
import { cosineSimilarity, createRecognitionSession, embedAlignedFace, extractFaceEmbedding, RECOGNITION_MODELS, GHOSTFACENET_MODEL_ID } from "../ghostFaceNet";
import type { RecognitionModelId } from "../ghostFaceNet";
import { checkFaceQuality } from "../faceQuality";
import { equalErrorRate, thresholdForTargetFar, thresholdMetrics, type LabeledComparison } from "../verificationMetrics";

const STORAGE_KEY = "visage.registered-faces.v3";
const DEFAULT_MATCH_THRESHOLD = 0.72;
type DetectorId = "yunet" | "scrfd" | "blazeface" | "mediapipe";
type RegisteredFace = { id: string; name: string; embedding: number[]; createdAt: string; model: RecognitionModelId; preprocessing: string };
type RecognizedFace = { face: DetectedFace; name: string; score: number; reason?: string };
type ComparisonResult = { score: number; decision: string; backend: string; enrollmentImage: string; verificationImage: string; stages: { enrollmentDetection: number; verificationDetection: number; enrollmentLandmarkInference: number; verificationLandmarkInference: number; enrollmentAlignment: number; verificationAlignment: number; enrollmentQuality: number; verificationQuality: number; enrollmentEmbedding: number; verificationEmbedding: number; comparison: number; total: number } };
const DETECTORS: Record<DetectorId, string> = { yunet: "YuNet", scrfd: "SCRFD-500M", blazeface: "MediaPipe BlazeFace Short Range", mediapipe: "BlazeFace + 468 landmarks" };

async function embedding(session: ort.InferenceSession, image: ImageData, face: DetectedFace, model: RecognitionModelId) {
  return (await extractFaceEmbedding(session, image, face, model)).embedding;
}

function annotate(context: CanvasRenderingContext2D, faces: RecognizedFace[]) {
  drawFaces(context, faces.map((item) => item.face));
  const scale = Math.max(.75, Math.min(context.canvas.width, context.canvas.height) / 480);
  context.save(); context.font = `600 ${13 * scale}px ui-monospace, monospace`; context.textBaseline = "middle";
  for (const item of faces) {
    const label = `${item.name} ยท ${Math.round(item.score * 100)}%`, w = context.measureText(label).width + 14 * scale, h = 22 * scale;
    const x = Math.max(0, Math.min(item.face.x, context.canvas.width - w));
    const y = item.face.y >= h + 4 * scale ? item.face.y - h - 4 * scale : item.face.y + item.face.height + 4 * scale;
    context.fillStyle = "rgba(15,25,31,.9)"; context.beginPath(); context.roundRect(x, y, w, h, 6 * scale); context.fill();
    context.fillStyle = item.name === "Unknown" ? "#ffd58a" : "#8dffdc"; context.fillText(label, x + 7 * scale, y + h / 2);
  }
  context.restore();
}

export function FaceRecognition({ detectors }: { detectors: Record<DetectorId, FaceDetector> }) {
  const videoRef = useRef<HTMLVideoElement>(null), overlayRef = useRef<HTMLCanvasElement>(null), frozenRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null), sessionRef = useRef<ort.InferenceSession | null>(null);
  const activeRef = useRef(false), generationRef = useRef(0), inferenceBusyRef = useRef(false);
  const [detectorId, setDetectorId] = useState<DetectorId>("yunet"), [mode, setMode] = useState<"freeze" | "live">("freeze");
  const [recognitionModel, setRecognitionModel] = useState<RecognitionModelId>(GHOSTFACENET_MODEL_ID);
  const [identificationThreshold, setIdentificationThreshold] = useState(DEFAULT_MATCH_THRESHOLD);
  const [verificationThreshold, setVerificationThreshold] = useState(DEFAULT_MATCH_THRESHOLD);
  const [backendPreference, setBackendPreference] = useState<"auto" | "wasm" | "webgpu">("auto");
  const [active, setActive] = useState(false), [status, setStatus] = useState("Ready to recognize faces");
  const [recognitionBackend, setRecognitionBackend] = useState("Not loaded");
  const [registered] = useState<RegisteredFace[]>(() => { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]"); } catch { return []; } });
  const [result, setResult] = useState<RecognizedFace[]>([]);
  const [benchmark, setBenchmark] = useState<{ average: number; median: number; p95: number; minimum: number; sameFaceCount: number; consistency: number; elapsedMs: number } | null>(null);
  const [benchmarking, setBenchmarking] = useState(false);
  const [enrollmentFile, setEnrollmentFile] = useState<File | null>(null);
  const [verificationFile, setVerificationFile] = useState<File | null>(null);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [comparisonError, setComparisonError] = useState("");
  const [comparing, setComparing] = useState(false);
  const [groundTruth, setGroundTruth] = useState<"genuine" | "impostor">("genuine");
  const [consentGiven, setConsentGiven] = useState(false);
  const [evaluationPairs, setEvaluationPairs] = useState<LabeledComparison[]>([]);
  const [targetFar, setTargetFar] = useState(0.01);

  function stop(message?: string) {
    activeRef.current = false; generationRef.current++;
    streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false); if (message) setStatus(message);
  }
  useEffect(() => () => {
    activeRef.current = false; generationRef.current++;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    void (async () => {
      while (inferenceBusyRef.current) await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const session = sessionRef.current; sessionRef.current = null;
      await session?.release();
    })();
  }, []);

  async function bulkTest() {
    const video = videoRef.current, target = result[0]?.face;
    if (!activeRef.current || !video || video.readyState < 2 || !target || benchmarking) return;
    if (result[0]?.name === "Retake") { setStatus(result[0].reason ?? "Adjust the face quality before testing"); return; }
    const frameCanvas = document.createElement("canvas"); frameCanvas.width = video.videoWidth; frameCanvas.height = video.videoHeight;
    const frameContext = frameCanvas.getContext("2d", { willReadFrequently: true });
    if (!frameContext || !frameCanvas.width || !frameCanvas.height) { setStatus("Could not capture a frame for the bulk test"); return; }
    frameContext.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
    const frame = frameContext.getImageData(0, 0, frameCanvas.width, frameCanvas.height);
    const session = sessionRef.current;
    stop("Preparing recognition bulk test..."); setBenchmark(null); setBenchmarking(true);
    try {
      while (inferenceBusyRef.current) await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      if (!session) throw new Error("Recognition model is not ready");
      inferenceBusyRef.current = true;
      setStatus("Running 5 warm-up and 30 measured recognition runs...");
      for (let i = 0; i < 5; i++) await embedding(session, frame, target, recognitionModel);
      const expectedName = result[0].name;
      const benchmarkStartedAt = performance.now();
      let sameFaceCount = 0;
      const times: number[] = [];
      for (let i = 0; i < 30; i++) {
        const startedAt = performance.now();
        const vector = await embedding(session, frame, target, recognitionModel);
        times.push(performance.now() - startedAt);
        const ranked = registered.filter((person) => person.model === recognitionModel).map((person) => ({ person, score: cosineSimilarity(vector, person.embedding) })).sort((a, b) => b.score - a.score)[0];
        const name = ranked && ranked.score >= identificationThreshold ? ranked.person.name : "Unknown";
        if (expectedName !== "Unknown" && name === expectedName) sameFaceCount++;
      }
      const elapsedMs = performance.now() - benchmarkStartedAt;
      times.sort((a, b) => a - b);
      const average = times.reduce((sum, time) => sum + time, 0) / times.length;
      const stats = { average, median: (times[14] + times[15]) / 2, p95: times[Math.ceil(times.length * .95) - 1], minimum: times[0], sameFaceCount, consistency: (sameFaceCount / times.length) * 100, elapsedMs };
      setBenchmark(stats); setStatus("Recognition bulk test complete");
    } catch (error) { setStatus(`Bulk test failed: ${String(error)}`); }
    finally { inferenceBusyRef.current = false; setBenchmarking(false); }
  }

  async function compareImages() {
    if (!enrollmentFile || !verificationFile || activeRef.current || comparing || benchmarking) return;
    setComparing(true); setComparison(null); setComparisonError("");
    inferenceBusyRef.current = true;
    let enrollmentBitmap: ImageBitmap | null = null, verificationBitmap: ImageBitmap | null = null;
    try {
      const detector = detectors[detectorId];
      const loaded = sessionRef.current ? { session: sessionRef.current, backend: recognitionBackend } : await createRecognitionSession(recognitionModel, backendPreference);
      const session = loaded.session;
      sessionRef.current = session;
      setRecognitionBackend(loaded.backend);
      await detector.load();
      const loadImage = async (file: File) => {
        const bitmap = await createImageBitmap(file);
        const canvas = document.createElement("canvas"); canvas.width = bitmap.width; canvas.height = bitmap.height;
        const context = canvas.getContext("2d", { willReadFrequently: true });
        if (!context) throw new Error("Could not read comparison image");
        context.drawImage(bitmap, 0, 0);
        return { bitmap, image: context.getImageData(0, 0, canvas.width, canvas.height) };
      };
      const totalStart = performance.now();
      const a = await loadImage(enrollmentFile); enrollmentBitmap = a.bitmap;
      const aDetectionStart = performance.now(), aDetection = await detector.detect(a.image), enrollmentDetection = performance.now() - aDetectionStart;
      if (aDetection.faces.length !== 1) throw new Error(`Enrollment image must contain exactly one face; found ${aDetection.faces.length}`);
      const aQuality = checkFaceQuality(a.image, aDetection.faces[0], 1);
      if (!aQuality.acceptable) throw new Error(`Enrollment image quality: ${aQuality.issues.join("; ")}`);
      if (!aQuality.alignedImage) throw new Error("Enrollment face alignment failed");
      const aEmbeddingStart = performance.now(), aVector = await embedAlignedFace(session, aQuality.alignedImage, recognitionModel), enrollmentEmbedding = performance.now() - aEmbeddingStart;

      const b = await loadImage(verificationFile); verificationBitmap = b.bitmap;
      const bDetectionStart = performance.now(), bDetection = await detector.detect(b.image), verificationDetection = performance.now() - bDetectionStart;
      if (bDetection.faces.length !== 1) throw new Error(`Verification image must contain exactly one face; found ${bDetection.faces.length}`);
      const bQuality = checkFaceQuality(b.image, bDetection.faces[0], 1);
      if (!bQuality.acceptable) throw new Error(`Verification image quality: ${bQuality.issues.join("; ")}`);
      if (!bQuality.alignedImage) throw new Error("Verification face alignment failed");
      const bEmbeddingStart = performance.now(), bVector = await embedAlignedFace(session, bQuality.alignedImage, recognitionModel), verificationEmbedding = performance.now() - bEmbeddingStart;
      if (aVector.length !== 512 || bVector.length !== 512) throw new Error("Comparison embeddings do not match the selected model's 512-value format");
      const comparisonStart = performance.now(), score = cosineSimilarity(aVector, bVector), comparisonMs = performance.now() - comparisonStart;
      const decision = score >= verificationThreshold ? "Match" : "No match";
      setComparison({
        score, decision, backend: loaded.backend,
        enrollmentImage: aQuality.alignedImage.toDataURL("image/png"),
        verificationImage: bQuality.alignedImage.toDataURL("image/png"),
        stages: { enrollmentDetection, verificationDetection, enrollmentLandmarkInference: aDetection.timings.landmarkInferenceMs ?? 0, verificationLandmarkInference: bDetection.timings.landmarkInferenceMs ?? 0, enrollmentAlignment: aQuality.alignmentMs, verificationAlignment: bQuality.alignmentMs, enrollmentQuality: aQuality.qualityMs, verificationQuality: bQuality.qualityMs, enrollmentEmbedding, verificationEmbedding, comparison: comparisonMs, total: performance.now() - totalStart },
      });
    } catch (error) { setComparisonError(String(error)); }
    finally { enrollmentBitmap?.close(); verificationBitmap?.close(); inferenceBusyRef.current = false; setComparing(false); }
  }

  const currentThresholdMetrics = thresholdMetrics(evaluationPairs, verificationThreshold);
  const selectedModelRegistered = registered.filter((person) => person.model === recognitionModel);
  const eer = equalErrorRate(evaluationPairs);
  const targetThreshold = thresholdForTargetFar(evaluationPairs, targetFar);
  const histogram = Array.from({ length: 10 }, (_, index) => {
    const lower = -1 + index * 0.2, upper = lower + 0.2;
    return {
      lower, upper,
      genuine: evaluationPairs.filter((pair) => pair.genuine && pair.score >= lower && (index === 9 ? pair.score <= upper : pair.score < upper)).length,
      impostor: evaluationPairs.filter((pair) => !pair.genuine && pair.score >= lower && (index === 9 ? pair.score <= upper : pair.score < upper)).length,
    };
  });

  async function start(nextMode: "freeze" | "live") {
    if (activeRef.current) return;
    const generation = ++generationRef.current; activeRef.current = true; setActive(true); setMode(nextMode); setResult([]); setBenchmark(null);
    setStatus("Starting camera...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
      if (!activeRef.current || generation !== generationRef.current) { stream.getTracks().forEach((track) => track.stop()); return; }
      streamRef.current = stream; const video = videoRef.current; if (!video) throw new Error("Camera unavailable");
      video.srcObject = stream; await video.play();
      const detector = detectors[detectorId];
      const hadSession = !!sessionRef.current;
      await detector.load();
      const loaded = sessionRef.current ? { session: sessionRef.current, backend: recognitionBackend } : await createRecognitionSession(recognitionModel, backendPreference);
      const session = loaded.session;
      sessionRef.current = session;
      setRecognitionBackend(loaded.backend);
      if (!activeRef.current || generation !== generationRef.current) {
        if (!hadSession) { await session.release(); if (sessionRef.current === session) sessionRef.current = null; }
        return;
      }
      const canvas = document.createElement("canvas"), ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) throw new Error("Could not capture camera frame");
      let previous: DetectedFace[] = [];
      setStatus(selectedModelRegistered.length ? "Looking for a registered face..." : `No faces registered with ${RECOGNITION_MODELS[recognitionModel].label}. Detected faces will show as Unknown.`);
      while (activeRef.current && generation === generationRef.current) {
        if (video.readyState >= 2) {
          canvas.width = video.videoWidth; canvas.height = video.videoHeight; ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          inferenceBusyRef.current = true;
          try {
          const frame = ctx.getImageData(0, 0, canvas.width, canvas.height), { faces } = await detector.detect(frame);
          if (!activeRef.current || generation !== generationRef.current) return;
          const smooth = smoothFaces(previous, faces); previous = smooth;
          const recognized: RecognizedFace[] = [];
          for (const face of smooth) {
            const quality = checkFaceQuality(frame, face, 1);
            if (!quality.acceptable) {
              recognized.push({ face, name: "Retake", score: 0, reason: quality.issues[0] });
              continue;
            }
            const vector = quality.alignedImage ? await embedAlignedFace(session, quality.alignedImage, recognitionModel) : await embedding(session, frame, face, recognitionModel);
            const ranked = registered.filter((person) => person.model === recognitionModel).map((person) => ({ person, score: cosineSimilarity(vector, person.embedding) })).sort((a, b) => b.score - a.score);
            const best = ranked[0], second = ranked[1];
            const ambiguous = best && second && best.score - second.score < 0.04;
            const match = best && best.score >= identificationThreshold && !ambiguous ? best.person.name : "Unknown";
            recognized.push({ face, name: match, score: best?.score ?? 0, reason: ambiguous ? "Possible match is ambiguous" : undefined });
          }
          if (!activeRef.current || generation !== generationRef.current) return;
          setResult(recognized);
          if (nextMode === "freeze" && recognized.length) {
            const output = frozenRef.current;
            if (output) { output.width = canvas.width; output.height = canvas.height; const outputCtx = output.getContext("2d"); if (outputCtx) { outputCtx.putImageData(frame, 0, 0); annotate(outputCtx, recognized); } }
            stop(recognized[0].name === "Unknown" ? "Face detected; no matching registered name." : `Recognized ${recognized[0].name}`);
            return;
          }
          const overlay = overlayRef.current;
          if (overlay) { overlay.width = canvas.width; overlay.height = canvas.height; const overlayCtx = overlay.getContext("2d"); if (overlayCtx) { overlayCtx.clearRect(0, 0, canvas.width, canvas.height); annotate(overlayCtx, recognized); } }
          setStatus(recognized.length ? `Identified ${recognized.filter((face) => face.name !== "Unknown" && face.name !== "Retake").length} of ${recognized.length} face(s)` : "Looking for faces...");
          } finally { inferenceBusyRef.current = false; }
        }
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      }
    } catch (error) { if (generation === generationRef.current) { stop(`Recognition error: ${String(error)}`); } }
  }

  return <div className="page-content register-content"><div className="view-intro"><div><p className="eyebrow">IDENTITY WORKSPACE</p><h2>Recognize registered faces</h2><p>Match the camera image against names saved in your local face register.</p></div><span className="privacy-chip"><span className="online-dot" /> PRIVATE BY DESIGN</span></div>
    <div className="recognition-layout">
    <section className="camera-panel register-camera" aria-labelledby="recognition-camera-title">
      <div className="camera-toolbar"><div className="camera-heading"><h2 id="recognition-camera-title">Face recognition</h2><span className={`live-indicator ${active ? "is-live" : ""}`}>{active ? "LIVE" : "READY"}</span></div>
        <div className="camera-controls"><label className="register-select-label">Detection model<select value={detectorId} disabled={active || comparing || benchmarking} onChange={(event) => setDetectorId(event.target.value as DetectorId)}>{Object.entries(DETECTORS).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label><label className="register-select-label">Recognition model<select value={recognitionModel} disabled={active || comparing || benchmarking} onChange={(event) => { const nextModel = event.target.value as RecognitionModelId; const oldSession = sessionRef.current; sessionRef.current = null; setRecognitionModel(nextModel); setRecognitionBackend("Not loaded"); setResult([]); setBenchmark(null); void oldSession?.release(); }}>{Object.entries(RECOGNITION_MODELS).map(([id, model]) => <option key={id} value={id}>{model.label}</option>)}</select></label><label className="register-select-label">Inference backend<select value={backendPreference} disabled={active || comparing || benchmarking} onChange={async (event) => { await sessionRef.current?.release(); sessionRef.current = null; setRecognitionBackend("Not loaded"); setBackendPreference(event.target.value as "auto" | "wasm" | "webgpu"); }}><option value="auto">Auto: WebGPU with WASM fallback</option><option value="webgpu">Prefer WebGPU</option><option value="wasm">WASM only</option></select></label><label className="register-select-label">Identification threshold<input type="number" min="0" max="1" step="0.01" value={identificationThreshold} onChange={(event) => setIdentificationThreshold(Math.min(1, Math.max(0, Number(event.target.value))))} /></label>
          {active ? <><button className="button button-benchmark" onClick={() => void bulkTest()} disabled={!result.length || result[0]?.name === "Retake" || benchmarking}>{benchmarking ? "Testing..." : "Bulk test"}</button><button className="button button-secondary" onClick={() => stop("Recognition stopped")} disabled={benchmarking}>Stop</button></> : <><button className="button button-primary" onClick={() => void start("freeze")} disabled={benchmarking || comparing}><span aria-hidden="true">โ—</span> Detect and freeze</button><button className="button button-secondary" onClick={() => void start("live")} disabled={benchmarking || comparing}>Live camera</button></>}
        </div>
      </div>
      <div className={`camera-stage ${active || result.length ? "camera-active" : ""}`}><video ref={videoRef} autoPlay muted playsInline className={`camera-media ${active ? "visible" : ""}`} /><canvas ref={overlayRef} className={`camera-overlay ${active && mode === "live" ? "visible" : ""}`} /><canvas ref={frozenRef} className={`camera-frozen ${!active && result.length && mode === "freeze" ? "visible" : ""}`} />
        {!active && !result.length && <div className="camera-empty"><strong>No frame</strong><span>Choose a mode to start recognition.</span></div>}
        <div className="camera-stage-footer"><span><span className={`status-dot ${active ? "is-live" : ""}`} />{status}</span><span>Camera frames stay on this device</span></div>
      </div>
    </section>
    <section className="registered-panel recognition-details-panel" aria-labelledby="recognition-results-heading">
      <div className="registered-heading"><div><h2 id="recognition-results-heading">Recognition results</h2><p>{RECOGNITION_MODELS[recognitionModel].label} · {recognitionBackend}</p></div><span className="face-count">{selectedModelRegistered.length} enrolled</span></div>
      <div className="recognition-summary"><strong>Identification threshold: {Math.round(identificationThreshold * 100)}%</strong><span>Names come from Face register. Only enrollments for this model are matched.</span>
        {result.length ? result.map((item, i) => <span key={`${item.name}-${i}`} className="recognition-name">{item.name}{item.name !== "Unknown" ? ` · ${(item.score * 100).toFixed(1)}%` : ""}</span>) : <span className="recognition-empty-result">Recognition results will appear here.</span>}
      </div>
      {benchmark && <div className="recognition-benchmark" aria-live="polite"><strong>Bulk test · {recognitionBackend}</strong><span>30 measured runs; 5 warm-ups excluded.</span><div><span>Same face recognized <b>{benchmark.sameFaceCount}/30 · {benchmark.consistency.toFixed(1)}%</b></span><span>Total test time <b>{benchmark.elapsedMs.toFixed(1)} ms</b></span><span>Average per run <b>{benchmark.average.toFixed(1)} ms</b></span><span>Median <b>{benchmark.median.toFixed(1)} ms</b></span><span>95th percentile <b>{benchmark.p95.toFixed(1)} ms</b></span><span>Fastest <b>{benchmark.minimum.toFixed(1)} ms</b></span></div></div>}
    </section>
    </div>
    <section className="camera-panel comparison-panel" aria-labelledby="comparison-title">
      <div className="section-heading"><div><h2 id="comparison-title">Compare two images</h2><p>Use fixed images to repeat 1:1 comparisons without changing camera frames. Images stay in memory and are not saved.</p></div></div>
      <div className="comparison-controls">
        <label>Enrollment image<input type="file" accept="image/*" disabled={active || comparing || benchmarking} onChange={(event) => setEnrollmentFile(event.target.files?.[0] ?? null)} /></label>
        <label>Verification image<input type="file" accept="image/*" disabled={active || comparing || benchmarking} onChange={(event) => setVerificationFile(event.target.files?.[0] ?? null)} /></label>
        <button className="button button-primary" onClick={() => void compareImages()} disabled={!enrollmentFile || !verificationFile || active || comparing || benchmarking}>{comparing ? "Comparing..." : "Compare images"}</button>
      </div>
      {comparisonError && <p className="comparison-error" role="alert">{comparisonError}</p>}
      {comparison && <div className="comparison-result" aria-live="polite">
        <div className="comparison-images"><figure><img src={comparison.enrollmentImage} alt="Aligned enrollment face at 112 by 112 pixels" /><figcaption>Enrollment · aligned 112 × 112</figcaption></figure><figure><img src={comparison.verificationImage} alt="Aligned verification face at 112 by 112 pixels" /><figcaption>Verification · aligned 112 × 112</figcaption></figure></div>
        <div className="comparison-decision"><span>Raw cosine similarity <strong>{comparison.score.toFixed(6)}</strong></span><span>Verification threshold <strong>{verificationThreshold.toFixed(2)}</strong></span><span>Decision <strong>{comparison.decision}</strong></span><span>Backend <strong>{comparison.backend}</strong></span></div>
        <div className="comparison-timings"><span>Detection A <b>{comparison.stages.enrollmentDetection.toFixed(1)} ms</b></span><span>Detection B <b>{comparison.stages.verificationDetection.toFixed(1)} ms</b></span><span>Landmarks A <b>{comparison.stages.enrollmentLandmarkInference.toFixed(1)} ms</b></span><span>Landmarks B <b>{comparison.stages.verificationLandmarkInference.toFixed(1)} ms</b></span><span>Alignment A <b>{comparison.stages.enrollmentAlignment.toFixed(1)} ms</b></span><span>Alignment B <b>{comparison.stages.verificationAlignment.toFixed(1)} ms</b></span><span>Quality A <b>{comparison.stages.enrollmentQuality.toFixed(1)} ms</b></span><span>Quality B <b>{comparison.stages.verificationQuality.toFixed(1)} ms</b></span><span>Embedding A <b>{comparison.stages.enrollmentEmbedding.toFixed(1)} ms</b></span><span>Embedding B <b>{comparison.stages.verificationEmbedding.toFixed(1)} ms</b></span><span>Comparison <b>{comparison.stages.comparison.toFixed(3)} ms</b></span><span>Total <b>{comparison.stages.total.toFixed(1)} ms</b></span></div>
      </div>}
      <div className="evaluation-controls">
        <label>Known relationship<select value={groundTruth} onChange={(event) => setGroundTruth(event.target.value as "genuine" | "impostor")}><option value="genuine">Same consenting person (genuine)</option><option value="impostor">Different people (impostor)</option></select></label>
        <label className="evaluation-consent"><input type="checkbox" checked={consentGiven} onChange={(event) => setConsentGiven(event.target.checked)} /> I have consent from the people shown to use this comparison for local evaluation.</label>
        <button className="button button-secondary" onClick={() => setEvaluationPairs((pairs) => [...pairs, { score: comparison!.score, genuine: groundTruth === "genuine" }])} disabled={!comparison || !consentGiven}>Add labeled comparison</button>
        {evaluationPairs.length > 0 && <button className="remove-face" onClick={() => setEvaluationPairs([])}>Clear evaluation set</button>}
      </div>
      {evaluationPairs.length > 0 && <div className="evaluation-results">
        <div className="evaluation-stats"><span>Pairs <b>{evaluationPairs.length}</b></span><span>Genuine <b>{currentThresholdMetrics.genuineCount}</b></span><span>Impostor <b>{currentThresholdMetrics.impostorCount}</b></span><span>FAR @ {verificationThreshold.toFixed(2)} <b>{currentThresholdMetrics.far === null ? "need impostor pairs" : `${(currentThresholdMetrics.far * 100).toFixed(2)}%`}</b></span><span>FRR @ {verificationThreshold.toFixed(2)} <b>{currentThresholdMetrics.frr === null ? "need genuine pairs" : `${(currentThresholdMetrics.frr * 100).toFixed(2)}%`}</b></span><span>EER <b>{eer ? `${(eer.eer * 100).toFixed(2)}% @ ${eer.threshold.toFixed(3)}` : "need both pair types"}</b></span></div>
        <div className="evaluation-controls"><label>Target FAR<select value={targetFar} onChange={(event) => setTargetFar(Number(event.target.value))}><option value={0.001}>0.1%</option><option value={0.01}>1%</option><option value={0.05}>5%</option></select></label><span>{targetThreshold ? `Validation threshold: ${targetThreshold.threshold.toFixed(3)} · FAR ${(targetThreshold.far! * 100).toFixed(2)}% · FRR ${(targetThreshold.frr! * 100).toFixed(2)}%` : "Add both genuine and impostor pairs to estimate a threshold."}</span>{targetThreshold && <button className="button button-secondary" onClick={() => setVerificationThreshold(targetThreshold.threshold)}>Use validation threshold</button>}</div>
        <div className="similarity-histogram" aria-label="Similarity score distributions">
          <div className="histogram-legend"><span>Genuine comparisons</span><span>Impostor comparisons</span></div>
          {histogram.map((bin) => <div className="histogram-row" key={bin.lower}><span>{bin.lower.toFixed(1)}–{bin.upper.toFixed(1)}</span><i className="genuine-bar" style={{ width: `${bin.genuine ? Math.max(2, bin.genuine / Math.max(1, ...histogram.flatMap((entry) => [entry.genuine, entry.impostor])) * 100) : 0}%` }} /><i className="impostor-bar" style={{ width: `${bin.impostor ? Math.max(2, bin.impostor / Math.max(1, ...histogram.flatMap((entry) => [entry.genuine, entry.impostor])) * 100) : 0}%` }} /><small>{bin.genuine}/{bin.impostor}</small></div>)}
        </div>
        <p className="evaluation-note">These rates describe only this small, user-labeled sample. They are not population performance estimates. The evaluation set is held in memory only.</p>
      </div>}
    </section>
  </div>;
}









