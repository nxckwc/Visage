import { describe, expect, it } from "vitest";
import { equalErrorRate, thresholdForTargetFar, thresholdMetrics } from "../src/verificationMetrics";

describe("recognition threshold metrics", () => {
  const sample = [
    { score: .9, genuine: true }, { score: .8, genuine: true },
    { score: .4, genuine: false }, { score: .2, genuine: false },
  ];

  it("reports FAR and FRR independently at a threshold", () => {
    expect(thresholdMetrics(sample, .5)).toMatchObject({ far: 0, frr: 0, genuineCount: 2, impostorCount: 2 });
    expect(thresholdMetrics(sample, .85)).toMatchObject({ far: 0, frr: .5 });
  });

  it("derives an EER operating point when both classes are present", () => {
    expect(equalErrorRate(sample)).toMatchObject({ far: 0, frr: 0, eer: 0 });
    expect(equalErrorRate([{ score: .5, genuine: true }])).toBeNull();
  });

  it("selects the lowest-FRR threshold that meets the requested FAR", () => {
    expect(thresholdForTargetFar(sample, 0)).toMatchObject({ threshold: .8, far: 0, frr: 0 });
    expect(thresholdForTargetFar([{ score: .5, genuine: true }], .01)).toBeNull();
  });
});
