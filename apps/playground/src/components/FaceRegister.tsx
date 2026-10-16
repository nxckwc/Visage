import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import type { FaceDetector, DetectedFace } from "../detectors/types";
import { cosineSimilarity, createRecognitionSession, embedAlignedFace, RECOGNITION_MODELS, GHOSTFACENET_MODEL_ID } from "../ghostFaceNet";
import type { RecognitionModelId } from "../ghostFaceNet";
import { checkFaceQuality } from "../faceQuality";

const STORAGE_KEY = "visage.registered-faces.v3";
type RegisteredFace = { id: string; name: string; embedding: number[]; createdAt: string; model: RecognitionModelId; preprocessing: string };
type PendingFace = { embedding: number[]; image: string };
type DetectorId = "yunet" | "scrfd" | "blazeface" | "mediapipe";

async function makeEmbedding(session: ort.InferenceSession, crop: HTMLCanvasElement, model: RecognitionModelId) {
  return embedAlignedFace(session, crop, model);
}

export function FaceRegister({ detectors }: { detectors: Record<DetectorId, FaceDetector> }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const runningRef = useRef(false);
  const generationRef = useRef(0);
  const inferenceBusyRef = useRef(false);
  const recognitionRef = useRef<ort.InferenceSession | null>(null);
  const [detectorId, setDetectorId] = useState<DetectorId>("yunet");
  const [recognitionModel, setRecognitionModel] = useState<RecognitionModelId>(GHOSTFACENET_MODEL_ID);
  const [backendPreference, setBackendPreference] = useState<"auto" | "wasm" | "webgpu">("auto");
  const [faces, setFaces] = useState<RegisteredFace[]>(() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as RegisteredFace[]; }
    catch { return []; }
  });
  const [pending, setPending] = useState<PendingFace | null>(null);
  const [replaceCandidate, setReplaceCandidate] = useState<RegisteredFace | null>(null);
  const [name, setName] = useState("");
  const [active, setActive] = useState(false);
  const [status, setStatus] = useState("Ready to register a face");
  const [loading, setLoading] = useState(false);
  const [recognitionBackend, setRecognitionBackend] = useState("Not loaded");

  function stop() {
    runningRef.current = false; generationRef.current++;
    streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
  }

  useEffect(() => () => {
    runningRef.current = false; generationRef.current++;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    void (async () => {
      while (inferenceBusyRef.current) await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const session = recognitionRef.current; recognitionRef.current = null;
      await session?.release();
    })();
  }, []);

  async function ensureRecognitionModel(generation: number) {
    if (recognitionRef.current) return recognitionRef.current;
    setStatus(`Loading ${RECOGNITION_MODELS[recognitionModel].label}${recognitionModel === "arcface_resnet100_v1" ? " (first load downloads about 249 MB)..." : "..."}`);
    const { session, backend } = await createRecognitionSession(recognitionModel, backendPreference);
    if (!runningRef.current || generation !== generationRef.current) {
      await session.release();
      throw new Error("Registration cancelled while loading the recognition model");
    }
    recognitionRef.current = session;
    setRecognitionBackend(backend);
    return session;
  }

  async function start() {
    if (active || loading) return;
    let generation = ++generationRef.current; runningRef.current = true; setActive(true); setLoading(true);
    try {
      const detector = detectors[detectorId];
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
      if (!runningRef.current || generation !== generationRef.current) { stream.getTracks().forEach((track) => track.stop()); return; }
      streamRef.current = stream;
      const video = videoRef.current; if (!video) throw new Error("Camera unavailable");
      video.srcObject = stream; await video.play();
      await detector.load();
      const recognitionSession = await ensureRecognitionModel(generation);
      setStatus("Looking for a face…");
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) throw new Error("Could not capture camera frame");
      let previousCandidate: DetectedFace | null = null;
      let stableFrames = 0;
      while (runningRef.current && generation === generationRef.current) {
        if (video.readyState >= 2) {
          if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
          if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0); const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const result = await detector.detect(frame);
          if (result.faces.length !== 1) {
            previousCandidate = null; stableFrames = 0;
            setStatus(result.faces.length ? "Only one face should be in view" : "Center one face in view");
          } else {
            const face = result.faces[0];
            const quality = checkFaceQuality(frame, face, result.faces.length);
            if (!quality.acceptable || !quality.alignedImage) {
              previousCandidate = null; stableFrames = 0;
              setStatus(quality.issues[0] ?? "Adjust position and lighting");
            } else {
              const previousOverlap = previousCandidate && (() => {
                const left = Math.max(previousCandidate!.x, face.x), top = Math.max(previousCandidate!.y, face.y);
                const right = Math.min(previousCandidate!.x + previousCandidate!.width, face.x + face.width);
                const bottom = Math.min(previousCandidate!.y + previousCandidate!.height, face.y + face.height);
                const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
                return intersection / (previousCandidate!.width * previousCandidate!.height + face.width * face.height - intersection);
              })();
              stableFrames = previousOverlap !== null && previousOverlap >= 0.6 ? stableFrames + 1 : 1;
              previousCandidate = face;
              if (stableFrames < 3) {
                setStatus(`Hold still while we check alignment (${stableFrames}/3)`);
              } else {
            stop();
            generation = generationRef.current;
            const recognitionGeneration = generation;
            const crop = quality.alignedImage;
            setStatus("Recognizing captured face…");
            const recognitionStartedAt = performance.now();
            const progressTimer = window.setInterval(() => {
              if (recognitionGeneration !== generationRef.current) return;
              const elapsedSeconds = Math.floor((performance.now() - recognitionStartedAt) / 1000);
              setStatus(elapsedSeconds < 20
                ? `Recognizing captured face… ${elapsedSeconds}s`
                : `Recognition is taking longer than expected (${elapsedSeconds}s). It may be slow on this device.`);
            }, 1000);
            let embedding: number[];
            inferenceBusyRef.current = true;
            try {
              embedding = await makeEmbedding(recognitionSession, crop, recognitionModel);
            } finally {
              inferenceBusyRef.current = false;
              window.clearInterval(progressTimer);
            }
            if (recognitionGeneration !== generationRef.current) return;
            setPending({ embedding, image: crop.toDataURL("image/jpeg", 0.88) });
            setStatus("Face captured. Add a name to register it.");
            return;
              }
            }
          }
        }
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      }
    } catch (error) {
      if (generation === generationRef.current) { stop(); setStatus(`Registration error: ${String(error)}`); }
    } finally { setLoading(false); }
  }

  function saveFace(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault(); if (!pending || (!replaceCandidate && !name.trim())) return;
    const match = faces.filter((face) => face.model === recognitionModel).map((face) => ({ face, score: cosineSimilarity(pending.embedding, face.embedding) })).sort((a, b) => b.score - a.score)[0];
    if (!replaceCandidate && match && match.score > 0.72) { setReplaceCandidate(match.face); setStatus(`This looks like ${match.face.name} (${(match.score * 100).toFixed(1)}% match). Review and replace its enrollment if this capture is better.`); return; }
    const now = new Date().toISOString();
    const next: RegisteredFace[] = replaceCandidate
      ? faces.map((face) => face.id === replaceCandidate.id ? { ...face, embedding: pending.embedding, createdAt: now, model: recognitionModel, preprocessing: RECOGNITION_MODELS[recognitionModel].preprocessing } : face)
      : [{ id: crypto.randomUUID(), name: name.trim(), embedding: pending.embedding, createdAt: now, model: recognitionModel, preprocessing: RECOGNITION_MODELS[recognitionModel].preprocessing }, ...faces];
    setFaces(next); localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    setPending(null); setName(""); setReplaceCandidate(null); setStatus(replaceCandidate ? `Enrollment replaced for ${replaceCandidate.name}` : "Face registered on this device");
  }

  function removeFace(id: string) {
    const next = faces.filter((face) => face.id !== id); setFaces(next); localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  }

  return <div className="register-layout">
    <section className="camera-panel register-camera" aria-labelledby="register-camera-title">
      <div className="camera-toolbar"><div className="camera-heading"><h2 id="register-camera-title">Register a face</h2><span className={`live-indicator ${active ? "is-live" : ""}`}>{active ? "LIVE" : "READY"}</span></div>
        <div className="camera-controls"><label className="register-select-label">Detection model<select value={detectorId} disabled={active || loading} onChange={(event) => setDetectorId(event.target.value as DetectorId)}>{Object.entries({ yunet: "YuNet", scrfd: "SCRFD-500M", blazeface: "MediaPipe BlazeFace Short Range", mediapipe: "BlazeFace + 468 landmarks" }).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
          <label className="register-select-label">Recognition model<select value={recognitionModel} disabled={active || loading} onChange={(event) => { const nextModel = event.target.value as RecognitionModelId; const oldSession = recognitionRef.current; recognitionRef.current = null; setRecognitionModel(nextModel); setRecognitionBackend("Not loaded"); void oldSession?.release(); }}>{Object.entries(RECOGNITION_MODELS).map(([id, model]) => <option key={id} value={id}>{model.label}</option>)}</select></label>
          <label className="register-select-label">Inference backend<select value={backendPreference} disabled={active || loading} onChange={async (event) => { await recognitionRef.current?.release(); recognitionRef.current = null; setRecognitionBackend("Not loaded"); setBackendPreference(event.target.value as "auto" | "wasm" | "webgpu"); }}><option value="auto">Auto: WebGPU with WASM fallback</option><option value="webgpu">Prefer WebGPU</option><option value="wasm">WASM only</option></select></label>
          {active ? <button className="button button-secondary" onClick={stop}>Stop</button> : loading ? <button className="button button-secondary" onClick={() => { stop(); setStatus("Cancelling registration…"); }}>Cancel</button> : <button className="button button-primary" onClick={() => void start()}><span aria-hidden="true">◉</span> Start registration</button>}
        </div>
      </div>
      <div className={`camera-stage ${active || pending ? "camera-active" : ""}`}>
        <video ref={videoRef} autoPlay muted playsInline className={`camera-media ${active ? "visible" : ""}`} />
        {pending && <div className="capture-review"><img src={pending.image} alt="Captured face crop" /><span>Face captured</span></div>}
        {!active && !pending && <div className="camera-empty"><strong>No frame</strong><span>Start registration and look toward the camera.</span></div>}
        <div className="camera-stage-footer"><span><span className={`status-dot ${active ? "is-live" : ""}`} />{status}</span><span>Camera frames stay on this device</span></div>
      </div>
      <div className="register-model-note"><span className="model-status-dot" /> {RECOGNITION_MODELS[recognitionModel].label} <span>·</span> {recognitionBackend} · 5-point aligned 112 × 112.</div>
    </section>

    <section className="registered-panel" aria-labelledby="registered-heading">
      <div className="registered-heading"><div><h2 id="registered-heading">Registered faces</h2><p>Names and face embeddings saved in this browser.</p></div><span className="face-count">{faces.length} {faces.length === 1 ? "face" : "faces"}</span></div>
      {faces.length ? <div className="registered-table-wrap"><table className="registered-table"><thead><tr><th>Person</th><th>Added</th><th></th></tr></thead><tbody>{faces.map((face) => <tr key={face.id}><td><span className="avatar-initial">{face.name.charAt(0).toUpperCase()}</span><strong>{face.name}</strong></td><td>{new Date(face.createdAt).toLocaleDateString()}</td><td><button className="remove-face" onClick={() => removeFace(face.id)} aria-label={`Remove ${face.name}`}>Remove</button></td></tr>)}</tbody></table></div> : <div className="registered-empty"><span className="empty-avatar">＋</span><strong>No faces registered yet</strong><span>Captured faces will appear here.</span></div>}
    </section>

    {pending && <div className="modal-backdrop" role="presentation"><form className="name-dialog" onSubmit={saveFace} aria-labelledby="name-dialog-title"><button className="dialog-close" type="button" aria-label="Close" onClick={() => { setPending(null); setReplaceCandidate(null); setStatus("Registration cancelled"); }}>×</button><span className="dialog-eyebrow">NEW FACE</span><h2 id="name-dialog-title">{replaceCandidate ? `Replace ${replaceCandidate.name}?` : "Who is this?"}</h2><p>{replaceCandidate ? "This capture matches an existing entry. Replace its saved face template with this capture?" : "Give this face a name to add it to your local register."}</p>{!replaceCandidate && <><label htmlFor="face-name">Name</label><input autoFocus id="face-name" value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. Alex Morgan" maxLength={80} required /></>}<div className="dialog-actions"><button className="button button-secondary" type="button" onClick={() => { setPending(null); setReplaceCandidate(null); setStatus("Registration cancelled"); }}>Cancel</button><button className="button button-primary" type="submit">{replaceCandidate ? "Replace enrollment" : "Save face"}</button></div></form></div>}
  </div>;
}
