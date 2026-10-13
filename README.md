# Visage

**Visage** is a TypeScript-first facial recognition SDK designed for fast, reusable, on-device face recognition in browsers and, eventually, Node.js applications.

The goal is to provide a framework-independent engine for detecting and aligning faces, extracting embeddings, and performing one-shot enrollment, verification, and identification. Integrations such as React and Next.js should build on the same underlying SDK rather than define its architecture.

## Goals

- **One-shot enrollment:** Create a face template from a single suitable image or video frame, without retraining a model for each person.
- **Low latency:** Optimize model loading, face detection, alignment, and recognition for responsive applications.
- **Browser-first, offline-capable inference:** Run locally where supported, without requiring face images to be uploaded to a server.
- **Reusable APIs:** Keep the recognition engine independent of React, Next.js, and other application frameworks.
- **Measurable reliability:** Benchmark accuracy, false matches, latency, memory usage, and model size before making performance claims.
- **Developer control:** Allow applications to choose their identity storage, matching thresholds, and deployment model.

## Repository structure

```text
visage/
├── apps/
│   └── playground/       # Vite + React development playground
├── packages/
│   ├── core/             # Framework-independent types and recognition logic
│   └── browser/          # Browser-specific SDK APIs and integrations
├── pnpm-workspace.yaml
├── pnpm-lock.yaml
└── package.json
```

| Workspace | Purpose | Current status |
| --- | --- | --- |
| `@visage/core` | Shared engine, types, and future recognition primitives | Scaffolded; exports SDK version |
| `@visage/browser` | Browser APIs, future image/camera and inference integration | Scaffolded; reports browser capabilities |
| `apps/playground` | Interactive environment for testing the SDK | Running Vite + React app |

Additional packages, such as `@visage/node` or framework-specific adapters, may be added after the core pipeline is validated.

## Getting started

### Requirements

- Node.js compatible with the installed Vite, TypeScript, and Vitest versions
- [pnpm](https://pnpm.io/)

### Installation

```bash
git clone https://github.com/nxckwc/visage.git
cd visage
pnpm install
```

### Start the playground

```bash
pnpm dev
```

Open the local URL printed by Vite (often `http://localhost:5173`). The current playground verifies that the browser package can load and access the core package. **It does not perform face recognition yet.**

### Build all workspaces

```bash
pnpm build
```

### Run tests and type checking

```bash
pnpm test
pnpm typecheck
```

The root workspace also provides a lint command:

```bash
pnpm lint
```

> The lint command runs scripts that exist in individual workspaces; not every package necessarily has a lint script configured yet.

## Current development milestone

**Foundation:** Set up the pnpm monorepo, `@visage/core`, `@visage/browser`, the playground, TypeScript builds, and an initial Vitest test.

**Next:** Evaluate browser-compatible face detectors, then implement face bounding boxes and safe image cropping. The detector should preferably expose five facial landmarks, enabling alignment without a separate heavy landmark pass.

## Planned recognition pipeline

```text
Image / Camera Frame
        |
        v
Face Detection + Bounding Boxes
        |
        v
Face Quality Check
        |
        v
Five-Point Landmark Alignment
        |
        v
Pretrained Face Embedding Model
        |
        v
Normalized Embedding
        |
        +--> Enroll: store template for an identity
        |
        +--> Verify: compare against a claimed identity (1:1)
        |
        +--> Identify: search an enrolled gallery (1:N)
```

This pipeline is **planned**, not a description of currently available functionality. Model selection and matching thresholds will be based on benchmark results.

## Roadmap

- [x] Initialize pnpm + TypeScript monorepo
- [x] Set up `@visage/core` and `@visage/browser`
- [x] Create Vite + React playground
- [x] Verify production builds and initial tests
- [ ] Benchmark and select a browser-compatible face detector
- [ ] Implement detection and face cropping
- [ ] Add five-point landmark alignment and quality checks
- [ ] Integrate a pretrained face-embedding model
- [ ] Implement one-shot enrollment and template storage interfaces
- [ ] Implement 1:1 verification and 1:N identification
- [ ] Benchmark latency, accuracy, and resource usage
- [ ] Add streaming support, Node.js integration, and release tooling


## Privacy and security

Face embeddings are **sensitive biometric data**, even when original photos are not retained. Applications using Visage will need appropriate user consent, secure storage, access controls, retention/deletion policies, and compliance with applicable law.

Face similarity alone **does not establish liveness** and is not sufficient as a standalone security measure for high-risk authentication. Photo and replay attacks, lookalikes, bias, and false matches must be assessed before deploying recognition in consequential settings.

## Contributing

Visage is in its initial development phase. Before changing an API or introducing a model dependency, document its compatibility, licensing, accuracy, latency, and bundle-size implications. Add tests for new behavior and use the playground to reproduce browser-specific issues.

## License

**Not specified yet.** Do not assume this repository or its future model weights are available under an open-source license until one is explicitly added. Third-party model weights may have separate licensing terms.
