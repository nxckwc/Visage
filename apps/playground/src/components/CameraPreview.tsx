import type { RefObject } from "react";

type CameraPreviewProps = {
  videoRef: RefObject<HTMLVideoElement | null>;
  resultCanvasRef: RefObject<HTMLCanvasElement | null>;
  isCapturing: boolean;
  hasResult: boolean;
};

export function CameraPreview({
  videoRef,
  resultCanvasRef,
  isCapturing,
  hasResult,
}: CameraPreviewProps) {
  return (
    <div style={{ marginTop: 16 }}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          display: isCapturing ? "block" : "none",
          width: "100%",
          borderRadius: 8,
        }}
      />
      <canvas
        ref={resultCanvasRef}
        style={{
          display: hasResult ? "block" : "none",
          width: "100%",
          borderRadius: 8,
        }}
      />
    </div>
  );
}
