import { describe, expect, it } from "vitest";
import { VERSION } from "./index.js";

describe("@visage/core", () => {
  it("exports the correct SDK version", () => {
    expect(VERSION).toBe("0.1.0");
  });
});