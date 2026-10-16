# Face recognition checklist status

## Implemented in the playground

- GhostFaceNet input is checked against the local ONNX graph: float32 NHWC `[1,112,112,3]`; the output must be float32 `[1,512]`. RGB pixels are scaled to `[-1,1]`, and embeddings are checked for finite values and L2-normalized. The exact file hash and the provenance caveat are recorded in `apps/playground/public/models/README.md`.
- Enrollment and verification use a shared five-point similarity transform into the 112px reference template. MediaPipe Face Mesh indices 33, 263, 1, 61, and 291 are used for its dense landmarks; other bundled detectors use their first five points. The same transform is used in both paths.
- Registration waits for one quality-approved face and three stable frames. It reports capture guidance, versions stored templates, supports replacing a matching enrollment, and supports deleting entries.
- Preliminary quality checks cover detector confidence, face dimensions, landmark geometry, coarse yaw/roll, blur, exposure, and clipped lighting. Enrollment rejects multiple faces. Live identification handles each detected face independently and rejects ambiguous top matches.
- The recognition page includes fixed-image 1:1 comparison, aligned 112×112 previews, raw similarity, separate verification threshold and decision, and stage timings. Labeled genuine/impostor scores can be collected in memory after a consent confirmation; the page plots their score distribution and calculates FAR, FRR, EER, and a threshold for a selected target FAR.
- Recognition inference can use Auto, WebGPU preference, or WASM only. A WASM fallback is used when WebGPU session creation is unavailable or fails.
- Geometry and metric tests are included under `apps/playground/tests`.

## Still needs evaluation data or additional assets

- The quality limits and identification threshold are provisional heuristics. The evaluation panel needs a meaningful, consented, labeled dataset across people, devices, lighting, and pose before the project can select a trustworthy threshold or report performance. The displayed FAR/FRR/EER only describe the local sample supplied by the operator.
- The local ONNX graph exposes tensor shapes but has no embedded checkpoint provenance. Its bytes are pinned by SHA-256, and the RGB `[-1,1]` scaling matches the upstream GhostFaceNets data pipeline; the exact weights and redistribution rights still need confirmation.
- SFace, MobileFaceNet, and a separately licensed reference recognizer have not been added. Their exact model files, licenses, accuracy datasets, and size/speed comparisons are prerequisites to a fair model selection.
- There is no liveness or presentation-attack model. Face similarity must not be treated as proof that a live person is present.
- Occlusion checks are limited to landmark geometry; the app does not reliably detect covered eyes, nose, or mouth. Yaw, pitch, blur, and exposure limits need measured calibration. Current capture heuristics may reject valid frames or accept poor ones.
- Embeddings are stored in browser `localStorage` as plaintext sensitive data. Deletion is supported, but encryption, retention controls, and access control are not implemented.
- Recognition and alignment remain in the playground. Stable framework-independent SDK APIs, worker-based inference, backend performance reports, package publication, and CI distribution documentation remain future work.

## Interpretation

Repeated runs on one frozen image measure numerical consistency and latency. They do not measure identity accuracy or robustness. Use separate images for genuine and impostor pairs, and evaluate conditions such as pose, lighting, device, expression, occlusion, and spoofing before relying on recognition decisions.
