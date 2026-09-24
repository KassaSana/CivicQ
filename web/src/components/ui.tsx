import { type ReactNode, useEffect, useRef, useState } from 'react';

export function SectionHead({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="section-head">
      <h2>{title}</h2>
      {children}
    </div>
  );
}

/** A numbered figure: content with a caption underneath. */
export function Figure({ n, caption, wide, className, children }: {
  n: number | string; caption: ReactNode; wide?: boolean; className?: string; children: ReactNode;
}) {
  return (
    <figure className={`fig ${wide ? 'wide' : ''} ${className ?? ''}`}>
      {children}
      <figcaption><b>Figure {n}.</b> {caption}</figcaption>
    </figure>
  );
}

export function Tile({ label, value, sub, variant, dim }: {
  label: string; value: ReactNode; sub?: ReactNode; variant?: 'flat' | 'bad'; dim?: boolean;
}) {
  return (
    <div className={`tile ${variant ?? ''} ${dim ? 'stale' : ''}`}>
      <div className="label">{label}</div>
      <div className="value num">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

/** Animates a number toward its target over ~180ms. */
export function useTween(target: number, ms = 180): number {
  const [v, setV] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    if (!Number.isFinite(target)) {
      setV(target);
      return;
    }
    const start = performance.now(), a = Number.isFinite(from.current) ? from.current : target;
    let raf = 0;
    const tick = (now: number) => {
      const k = Math.min(1, (now - start) / ms);
      const e = 1 - (1 - k) ** 3;
      const cur = a + (target - a) * e;
      from.current = cur;
      setV(cur);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

export function Num({ value, digits = 2, suffix = '' }: { value: number; digits?: number; suffix?: string }) {
  const v = useTween(value);
  return <>{Number.isFinite(v) ? v.toFixed(digits) : '∞'}{suffix}</>;
}

/** Hover tooltip positioned in viewBox coordinates over an SVG that fills its wrapper. */
export interface Tip { x: number; y: number; text: string }
export function useTip(vbW: number, vbH: number) {
  const [tip, setTip] = useState<Tip | null>(null);
  const node = tip ? (
    <div className="tooltip num" style={{ left: `${(tip.x / vbW) * 100}%`, top: `${(tip.y / vbH) * 100}%` }}>
      {tip.text}
    </div>
  ) : null;
  const bind = (x: number, y: number, text: string) => ({
    onMouseEnter: () => setTip({ x, y, text }),
    onMouseLeave: () => setTip(null),
  });
  return { node, bind };
}

export const pct = (p: number, d = 1) => `${(p * 100).toFixed(d)}%`;
