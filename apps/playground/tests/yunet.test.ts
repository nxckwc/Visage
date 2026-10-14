import { beforeEach, describe, expect, it, vi } from "vitest";
import * as ort from "onnxruntime-web";
import { YuNetDetector } from "../src/detectors/yunet";

vi.mock("onnxruntime-web", () => ({
  InferenceSession: { create: vi.fn() },
  Tensor: class {
    dispose = vi.fn();
  },
}));

const createSession = vi.mocked(ort.InferenceSession.create);

function emptyOutputs() {
  return Object.fromEntries(
    [8, 16, 32].flatMap((stride) =>
      ["cls", "obj", "bbox", "kps"].map((kind) => [
        `${kind}_${stride}`,
        { data: new Float32Array(0), dispose: vi.fn() },
      ])
    )
  ) as Record<string, ort.Tensor>;
}

function makeSession(run: () => Promise<Record<string, ort.Tensor>>) {
  return {
    run: vi.fn(run),
    release: vi.fn(async () => undefined),
  } as unknown as ort.InferenceSession;
}

function installCanvasMock() {
  const context = {
    fillStyle: "",
    fillRect: vi.fn(),
    drawImage: vi.fn(),
    putImageData: vi.fn(),
    getImageData: vi.fn(() => ({ data: new Uint8ClampedArray(640 * 640 * 4) })),
  };
  vi.stubGlobal("document", {
    createElement: vi.fn(() => ({
      width: 0,
      height: 0,
      getContext: vi.fn(() => context),
    })),
  });
}

const frame = {
  width: 2,
  height: 2,
  data: new Uint8ClampedArray(16),
} as ImageData;

describe("YuNetDetector lifecycle", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    installCanvasMock();
  });

  it("waits for active inference before releasing the session", async () => {
    let finishRun!: (outputs: Record<string, ort.Tensor>) => void;
    const session = makeSession(
      () => new Promise((resolve) => { finishRun = resolve; })
    );
    createSession.mockResolvedValue(session);

    const detector = new YuNetDetector();
    await detector.load();
    const detection = detector.detect(frame);
    const disposal = detector.dispose();

    expect(session.release).not.toHaveBeenCalled();
    finishRun(emptyOutputs());
    await detection;
    await disposal;

    expect(session.release).toHaveBeenCalledOnce();
  });

  it("rejects overlapping detect calls", async () => {
    let finishRun!: (outputs: Record<string, ort.Tensor>) => void;
    const session = makeSession(
      () => new Promise((resolve) => { finishRun = resolve; })
    );
    createSession.mockResolvedValue(session);

    const detector = new YuNetDetector();
    await detector.load();
    const detection = detector.detect(frame);

    await expect(detector.detect(frame)).rejects.toThrow("already in progress");
    finishRun(emptyOutputs());
    await detection;
    await detector.dispose();
  });

  it("rejects detection before load", async () => {
    const detector = new YuNetDetector();
    await expect(detector.detect(frame)).rejects.toThrow("must be loaded");
    await detector.dispose();
  });

  it("releases a session created while disposal is waiting for load", async () => {
    let finishLoad!: (session: ort.InferenceSession) => void;
    createSession.mockReturnValue(new Promise((resolve) => { finishLoad = resolve; }));
    const detector = new YuNetDetector();
    const loading = detector.load();
    const disposal = detector.dispose();
    const session = makeSession(async () => emptyOutputs());

    finishLoad(session);
    await expect(loading).rejects.toThrow("disposed during load");
    await disposal;
    expect(session.release).toHaveBeenCalledOnce();
  });

  it("makes load and dispose idempotent and forbids use after disposal", async () => {
    const session = makeSession(async () => emptyOutputs());
    createSession.mockResolvedValue(session);

    const detector = new YuNetDetector();
    await Promise.all([detector.load(), detector.load()]);
    expect(createSession).toHaveBeenCalledOnce();

    await Promise.all([detector.dispose(), detector.dispose()]);
    expect(session.release).toHaveBeenCalledOnce();
    await expect(detector.detect(frame)).rejects.toThrow("disposed");
  });
});
