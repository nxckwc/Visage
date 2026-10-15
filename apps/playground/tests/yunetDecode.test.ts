import { describe, expect, it } from "vitest";
import type * as ort from "onnxruntime-web";
import { decodeYuNet } from "../src/yunetDecode";

function outputsWithDetection(): Record<string, ort.Tensor> {
  return Object.fromEntries(
    [8, 16, 32].flatMap((stride) => {
      const length = stride === 8 ? 1 : 0;
      const cls = new Float32Array(length);
      const obj = new Float32Array(length);
      const bbox = new Float32Array(length * 4);
      const kps = new Float32Array(length * 10);
      if (stride === 8) {
        cls[0] = 1;
        obj[0] = 1;
        bbox.set([20, 20, Math.log(10), Math.log(10)]);
        kps.set([20, 20, 30, 20, 25, 25, 22, 30, 28, 30]);
      }
      return [
        [`cls_${stride}`, { data: cls }],
        [`obj_${stride}`, { data: obj }],
        [`bbox_${stride}`, { data: bbox }],
        [`kps_${stride}`, { data: kps }],
      ];
    })
  ) as Record<string, ort.Tensor>;
}

describe("decodeYuNet", () => {
  it("maps padded model geometry to source coordinates and clips landmarks", () => {
    const faces = decodeYuNet(
      outputsWithDetection(),
      {
        scale: 2,
        offsetX: 0,
        offsetY: 80,
        sourceWidth: 320,
        sourceHeight: 240,
      }
    );

    expect(faces).toHaveLength(1);
    expect(faces[0].x).toBeCloseTo(60);
    expect(faces[0].y).toBeCloseTo(20);
    expect(faces[0].width).toBeCloseTo(40);
    expect(faces[0].height).toBeCloseTo(40);
    expect(faces[0].confidence).toBe(1);
    expect(faces[0].landmarks[0]).toEqual({ x: 80, y: 40 });
    expect(faces[0].landmarks.every(({ x, y }) => x >= 0 && x <= 320 && y >= 0 && y <= 240)).toBe(true);
  });

  it("rejects malformed output tensor shapes", () => {
    const outputs = outputsWithDetection();
    outputs["bbox_8"] = { data: new Float32Array(1) } as ort.Tensor;

    expect(() => decodeYuNet(outputs, {
      scale: 1,
      offsetX: 0,
      offsetY: 0,
      sourceWidth: 640,
      sourceHeight: 640,
    })).toThrow("shapes do not match");
  });
});
