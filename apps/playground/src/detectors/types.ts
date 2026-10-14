export type DetectedFace = {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  landmarks: { x: number; y: number }[];
};

export type DetectorTimings = {
  preprocessingMs: number;
  inferenceMs: number;
  decodingMs: number;
};

export interface FaceDetector {
  readonly id: string;
  readonly name: string;
  readonly inputSize: { width: number; height: number };
  load(): Promise<void>;
  detect(image: ImageData): Promise<{
    faces: DetectedFace[];
    timings: DetectorTimings;
  }>;
  dispose(): Promise<void>;
}
