/**
 * Seeded random streams.
 *
 * Mirrors the C++ simulator's common-random-numbers design
 * (cpp/src/simulation.cpp, reset()): each day seed gets separate arrival,
 * service and daily-rate streams, so every staffing plan sees the same
 * citizens with the same service needs.
 */

export type Rng = () => number;

/** 32-bit mix of (seed, stream) so nearby seeds give unrelated streams. */
function hashSeed(seed: number, stream: number): number {
  let h = (seed ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  h ^= h >>> 16;
  h = (h + Math.imul(stream + 1, 0x27d4eb2f)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  return (h ^ (h >>> 15)) >>> 0;
}

/** mulberry32: small, fast, good enough for simulation. Returns U[0, 1). */
export function makeRng(seed: number, stream: number): Rng {
  let a = hashSeed(seed, stream);
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Exponential with the given mean. 1 - U is in (0, 1], so the log is finite. */
export function exponential(rng: Rng, mean: number): number {
  return -Math.log(1 - rng()) * mean;
}

/** Standard normal via Box-Muller. */
export function normal(rng: Rng): number {
  const u1 = 1 - rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Gamma(shape, scale) via Marsaglia-Tsang. */
export function gamma(rng: Rng, shape: number, scale: number): number {
  if (shape < 1) {
    // Boost: Gamma(a) = Gamma(a + 1) * U^(1/a)
    return gamma(rng, shape + 1, scale) * Math.pow(1 - rng(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number, v: number;
    do {
      x = normal(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = 1 - rng();
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v * scale;
  }
}
