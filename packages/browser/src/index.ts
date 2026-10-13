import { VERSION } from "@visage/core";

export function getBrowserInfo() {
  return {
    version: VERSION,
    webcamSupported:
      typeof navigator !== "undefined" &&
      !!navigator.mediaDevices?.getUserMedia
  };
}