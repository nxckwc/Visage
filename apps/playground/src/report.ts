export type TimingPoint = { x: number; y: number };

export function buildTimingPoints(
  values: number[],
  width: number,
  height: number,
): TimingPoint[] {
  if (values.length === 0) return [];

  const padding = 8;
  const min = Math.min(...values);
  const range = Math.max(...values) - min || 1;

  return values.map((value, index) => ({
    x: values.length === 1 ? width / 2 : (index / (values.length - 1)) * width,
    y: padding + (1 - (value - min) / range) * (height - padding * 2),
  }));
}
