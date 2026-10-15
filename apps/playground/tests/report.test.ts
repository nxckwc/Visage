import { describe, expect, it } from "vitest";
import { buildTimingPoints, buildTimingScale } from "../src/report";

describe("buildTimingPoints", () => {
  it("maps timing values across the chart bounds", () => {
    expect(buildTimingPoints([10, 20, 30], 100, 50)).toEqual([
      { x: 0, y: 42 },
      { x: 50, y: 25 },
      { x: 100, y: 8 },
    ]);
  });

  it("centers a single run and handles an empty report", () => {
    expect(buildTimingPoints([8], 100, 50)).toEqual([{ x: 50, y: 42 }]);
    expect(buildTimingPoints([], 100, 50)).toEqual([]);
  });
});
