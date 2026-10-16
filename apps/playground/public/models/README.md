# Face recognition model record

## GhostFaceNet V1 ONNX

- File: `ghostfacenet_v1.onnx`
- SHA-256: `77A2C233C29BF2CC99048C50EE8FAB018C8DD9DB2D0D1331A595A72668BF623D`
- Runtime-inspected input: float32 `[1, 112, 112, 3]` (NHWC)
- Runtime-inspected output: float32 `[1, 512]`
- Pixels are RGB and scaled from `[0,255]` to `[-1,1]`, matching the official GhostFaceNets training pipeline.
- Embeddings are checked for shape and finite values, then L2-normalized.
- Face alignment version: `arcface-5point-112-v1`; ArcFace five-point reference coordinates.

The ONNX graph has no embedded source/checkpoint provenance. The hash pins the artifact inspected locally, but its exact training weights and redistribution license are not established by the graph metadata. Treat model provenance and license validation as unresolved before redistribution.

## ArcFace ResNet100 (ONNX Model Zoo)

- Model: `arcfaceresnet100-8.onnx`, loaded lazily from the ONNX Model Zoo's Hugging Face mirror on first use.
- Source: `https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8`
- License: Apache-2.0 per the ONNX Model Zoo model card.
- Expected input: float32 NCHW `[1, 3, 112, 112]`, BGR pixels normalized to `[-1, 1]`.
- Expected output: float32 `[1, 512]`, L2-normalized by the app.
- The model card identifies this as LResNet100E-IR / ResNet100 and documents affine face alignment before inference.
- The full-precision model is approximately 249 MB; it is downloaded by ONNX Runtime Web when this model is first selected for use. Network/CORS access to Hugging Face is required.
- Existing GhostFaceNet enrollments are retained. Embeddings are model-specific, so recognition and duplicate checks only compare records enrolled with the selected model. Register the person again for ArcFace ResNet100.
