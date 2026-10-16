import type {
  DetectedFace,
  DetectorTimings,
  FaceDetector,
} from "./types";

const ALIGNED_SIZE = 256;
const CROP_SCALE = 1.5;
const MIN_ALIGNMENT_CONFIDENCE = 0.6;

type Alignment = {
  centerX: number;
  centerY: number;
  cropSize: number;
  rotation: number;
};

export class BlazeFaceLandmarkPipeline implements FaceDetector {
  readonly id = "mediapipe";
  readonly name = "BlazeFace 6-point → MediaPipe 468-point";
  readonly inputSize = { width: ALIGNED_SIZE, height: ALIGNED_SIZE };

  private loaded = false;
  private loadPromise: Promise<void> | null = null;
  private activeDetection: Promise<{
    faces: DetectedFace[];
    timings: DetectorTimings;
  }> | null = null;
  private disposePromise: Promise<void> | null = null;
  private disposed = false;
  private sourceCanvas: HTMLCanvasElement | null = null;
  private alignedCanvas: HTMLCanvasElement | null = null;
  private readonly aligner: FaceDetector;
  private readonly landmarker: FaceDetector;

  constructor(aligner: FaceDetector, landmarker: FaceDetector) {
    this.aligner = aligner;
    this.landmarker = landmarker;
  }

  load(): Promise<void> {
    if (this.disposed) {
      return Promise.reject(new Error("Landmark pipeline has been disposed"));
    }
    if (this.loaded) return Promise.resolve();
    if (this.loadPromise) return this.loadPromise;

    const loading = Promise.all([
      this.aligner.load(),
      this.landmarker.load(),
    ]).then(() => {
      if (this.disposed) {
        throw new Error("Landmark pipeline was disposed while loading");
      }
      this.loaded = true;
    });
    this.loadPromise = loading;
    void loading.then(
      () => {
        if (this.loadPromise === loading) this.loadPromise = null;
      },
      () => {
        if (this.loadPromise === loading) this.loadPromise = null;
      }
    );
    return loading;
  }

  detect(image: ImageData): Promise<{
    faces: DetectedFace[];
    timings: DetectorTimings;
  }> {
    if (this.disposed) {
      return Promise.reject(new Error("Landmark pipeline has been disposed"));
    }
    if (!this.loaded) {
      return Promise.reject(new Error("Landmark pipeline must be loaded before detect()"));
    }
    if (this.activeDetection) {
      return Promise.reject(new Error("Landmark pipeline is already processing a frame"));
    }

    const work = this.detectFrame(image);
    this.activeDetection = work;
    void work.then(
      () => {
        if (this.activeDetection === work) this.activeDetection = null;
      },
      () => {
        if (this.activeDetection === work) this.activeDetection = null;
      }
    );
    return work;
  }

  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;

    const disposal = (async () => {
      await this.loadPromise?.catch(() => undefined);
      await this.activeDetection?.catch(() => undefined);
      await Promise.all([this.aligner.dispose(), this.landmarker.dispose()]);
      this.loaded = false;
      this.sourceCanvas = null;
      this.alignedCanvas = null;
    })();
    this.disposePromise = disposal;
    return disposal;
  }

  private async detectFrame(image: ImageData) {
    const firstStage = await this.aligner.detect(image);
    const timings = { ...firstStage.timings, landmarkInferenceMs: 0 };
    const candidates = firstStage.faces.filter((face) =>
      Number.isFinite(face.confidence) && face.confidence >= MIN_ALIGNMENT_CONFIDENCE
    );
    if (candidates.length === 0) return { faces: [], timings };

    const faces: DetectedFace[] = [];
    for (const face of candidates) {
      const preprocessStartedAt = performance.now();
      const { image: alignedImage, alignment } = this.alignFace(image, face);
      timings.preprocessingMs += performance.now() - preprocessStartedAt;

      const refined = await this.landmarker.detect(alignedImage);
      timings.preprocessingMs += refined.timings.preprocessingMs;
      timings.inferenceMs += refined.timings.inferenceMs;
      timings.landmarkInferenceMs += refined.timings.inferenceMs;

      const decodeStartedAt = performance.now();
      const bestRefinement = refined.faces.reduce<DetectedFace | null>(
        (best, candidate) => !best || candidate.confidence > best.confidence
          ? candidate
          : best,
        null
      );
      if (bestRefinement) {
        faces.push({
          ...face,
          landmarks: bestRefinement.landmarks.map((point) =>
            this.mapPointToSource(point, alignment, image)
          ),
        });
      }
      timings.decodingMs += refined.timings.decodingMs + performance.now() - decodeStartedAt;
    }

    return { faces, timings };
  }

  private alignFace(image: ImageData, face: DetectedFace) {
    if (![face.x, face.y, face.width, face.height].every(Number.isFinite) ||
        face.width <= 0 || face.height <= 0) {
      throw new Error("BlazeFace returned invalid face bounds");
    }

    const eyeA = face.landmarks[0];
    const eyeB = face.landmarks[1];
    let rotation = eyeA && eyeB &&
      [eyeA.x, eyeA.y, eyeB.x, eyeB.y].every(Number.isFinite)
      ? Math.atan2(eyeB.y - eyeA.y, eyeB.x - eyeA.x)
      : 0;
    if (rotation > Math.PI / 2) rotation -= Math.PI;
    if (rotation < -Math.PI / 2) rotation += Math.PI;

    const alignment: Alignment = {
      centerX: face.x + face.width / 2,
      centerY: face.y + face.height / 2,
      cropSize: Math.max(face.width, face.height) * CROP_SCALE,
      rotation,
    };

    const source = this.sourceCanvas ??= document.createElement("canvas");
    if (source.width !== image.width) source.width = image.width;
    if (source.height !== image.height) source.height = image.height;
    const sourceContext = source.getContext("2d", { willReadFrequently: true });
    if (!sourceContext) throw new Error("Could not initialize alignment source canvas");
    sourceContext.putImageData(image, 0, 0);

    const canvas = this.alignedCanvas ??= document.createElement("canvas");
    if (canvas.width !== ALIGNED_SIZE) canvas.width = ALIGNED_SIZE;
    if (canvas.height !== ALIGNED_SIZE) canvas.height = ALIGNED_SIZE;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Could not initialize aligned face canvas");

    context.save();
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.fillStyle = "black";
    context.fillRect(0, 0, ALIGNED_SIZE, ALIGNED_SIZE);
    context.translate(ALIGNED_SIZE / 2, ALIGNED_SIZE / 2);
    context.rotate(-rotation);
    context.scale(ALIGNED_SIZE / alignment.cropSize, ALIGNED_SIZE / alignment.cropSize);
    context.drawImage(source, -alignment.centerX, -alignment.centerY);
    context.restore();

    return {
      image: context.getImageData(0, 0, ALIGNED_SIZE, ALIGNED_SIZE),
      alignment,
    };
  }

  private mapPointToSource(
    point: { x: number; y: number },
    alignment: Alignment,
    image: ImageData
  ) {
    const localX = (point.x - ALIGNED_SIZE / 2) * alignment.cropSize / ALIGNED_SIZE;
    const localY = (point.y - ALIGNED_SIZE / 2) * alignment.cropSize / ALIGNED_SIZE;
    const cosine = Math.cos(alignment.rotation);
    const sine = Math.sin(alignment.rotation);
    return {
      x: Math.max(0, Math.min(image.width, alignment.centerX + cosine * localX - sine * localY)),
      y: Math.max(0, Math.min(image.height, alignment.centerY + sine * localX + cosine * localY)),
    };
  }
}
