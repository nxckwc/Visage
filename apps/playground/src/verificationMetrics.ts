export type LabeledComparison = { score: number; genuine: boolean };
export type ThresholdMetrics = { threshold: number; far: number | null; frr: number | null; genuineCount: number; impostorCount: number };

export function thresholdMetrics(comparisons: LabeledComparison[], threshold: number): ThresholdMetrics {
  const genuine = comparisons.filter((pair) => pair.genuine);
  const impostor = comparisons.filter((pair) => !pair.genuine);
  const falseAccepts = impostor.filter((pair) => pair.score >= threshold).length;
  const falseRejects = genuine.filter((pair) => pair.score < threshold).length;
  return {
    threshold,
    far: impostor.length ? falseAccepts / impostor.length : null,
    frr: genuine.length ? falseRejects / genuine.length : null,
    genuineCount: genuine.length,
    impostorCount: impostor.length,
  };
}

export function equalErrorRate(comparisons: LabeledComparison[]) {
  const genuine = comparisons.filter((pair) => pair.genuine);
  const impostor = comparisons.filter((pair) => !pair.genuine);
  if (!genuine.length || !impostor.length) return null;
  const candidates = [...new Set(comparisons.map((pair) => pair.score))].sort((a, b) => a - b);
  let best: { threshold: number; far: number; frr: number; eer: number } | null = null;
  for (const threshold of [candidates[0] - Number.EPSILON, ...candidates]) {
    const metrics = thresholdMetrics(comparisons, threshold);
    const far = metrics.far!, frr = metrics.frr!;
    if (!best || Math.abs(far - frr) < Math.abs(best.far - best.frr)) best = { threshold, far, frr, eer: (far + frr) / 2 };
  }
  return best;
}

export function thresholdForTargetFar(comparisons: LabeledComparison[], targetFar: number) {
  const impostors = comparisons.filter((pair) => !pair.genuine);
  const genuine = comparisons.filter((pair) => pair.genuine);
  if (!impostors.length || !genuine.length) return null;
  const candidates = [...new Set(comparisons.map((pair) => pair.score))].sort((a, b) => a - b);
  const valid = candidates.map((threshold) => thresholdMetrics(comparisons, threshold))
    .filter((metrics) => metrics.far !== null && metrics.far <= targetFar);
  return valid.sort((a, b) => (a.frr ?? 1) - (b.frr ?? 1))[0] ?? null;
}
