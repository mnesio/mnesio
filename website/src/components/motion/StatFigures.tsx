/**
 * The proof strip — the second and last animated element on the page.
 *
 * Only the figures move, and only once: a count-up reads as the number being
 * *arrived at* rather than asserted, which is the right gesture for a project
 * whose argument is that it measures things. The cards themselves do not
 * animate, per the generated rule capping animation at 1-2 elements per view.
 *
 * Every figure is measured in this repository and reproducible from
 * `crates/mnesio-bench/manifest/`. Each carries the condition it was taken
 * under, including the one that is unflattering.
 */
import { useEffect, useRef, useState } from "react";
import { useInView, useReducedMotion } from "framer-motion";

type Figure = {
  value: number;
  decimals?: number;
  unit?: string;
  label: string;
  note: string;
};

const FIGURES: Figure[] = [
  {
    value: 4.8,
    decimals: 1,
    unit: "×",
    label: "fewer tokens at equal recall",
    note: "symbol context vs whole-file, paired on a single index",
  },
  {
    value: 4.11,
    decimals: 2,
    unit: "ms",
    label: "query latency, p99",
    note: "at 105,000 memories indexed",
  },
  {
    value: 73567,
    label: "symbols from one monorepo",
    note: "Kubernetes — 5.35M lines of Go across 13,420 files",
  },
  {
    value: 37.7,
    decimals: 1,
    unit: "min",
    label: "to cold-index that monorepo",
    note: "slow, and published anyway: comparable tools do 28M LOC in 3 min",
  },
  {
    value: 30,
    label: "languages parsed",
    note: "tree-sitter grammars — Rust, Python, TS, Go, Java, Swift, Zig…",
  },
  {
    value: 728,
    label: "tests in the workspace",
    note: "hard-rule invariants gate every release",
  },
];

/**
 * Count to `to` once, when the strip first scrolls into view.
 *
 * Starts at `null`, meaning "not animating", and `null` renders the *final*
 * value. That matters: the server-rendered HTML then contains 4.8x and 73,567
 * rather than a column of zeros, so a reader with JavaScript disabled — or one
 * who sees the page before hydration — gets the true numbers instead of false
 * ones. A count-up that degrades into misinformation would be a poor trade for
 * a page whose entire subject is measurement.
 */
function useCountUp(to: number, run: boolean) {
  const [n, setN] = useState<number | null>(null);
  const still = useReducedMotion();

  useEffect(() => {
    if (!run || still) return;
    // Never *start* a count in a document that is already hidden — rAF will
    // not tick, and the figure would sit frozen at whatever partial value the
    // first frame produced. `visibilitychange` below only fires on a
    // transition, so it cannot rescue an animation that began this way.
    if (typeof document !== "undefined" && document.hidden) {
      setN(to);
      return;
    }
    const DURATION = 900;
    const start = performance.now();
    let frame = requestAnimationFrame(function tick(now) {
      const t = Math.min(1, (now - start) / DURATION);
      // Ease-out cubic, so the last digits settle slowly enough to read.
      // The final frame is assigned exactly, never left on an eased
      // approximation of the value.
      setN(t < 1 ? to * (1 - Math.pow(1 - t, 3)) : to);
      if (t < 1) frame = requestAnimationFrame(tick);
    });

    // **The number on screen must never be a wrong number.** Browsers pause
    // rAF in a background tab, which strands a count-up wherever it happened
    // to be — this page rendered "0.2x fewer tokens" and "2,665 symbols" that
    // way, which is worse than showing no animation at all on a site whose
    // entire claim is that it measures rather than asserts. If the document
    // goes hidden, abandon the animation and commit the true value.
    const settle = () => {
      if (document.hidden) {
        cancelAnimationFrame(frame);
        setN(to);
      }
    };
    document.addEventListener("visibilitychange", settle);

    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("visibilitychange", settle);
    };
  }, [to, run, still]);

  return n ?? to;
}

function Figure({ f, run }: { f: Figure; run: boolean }) {
  const n = useCountUp(f.value, run);
  const shown =
    f.decimals != null
      ? n.toFixed(f.decimals)
      : Math.round(n).toLocaleString("en-US");

  return (
    <li className="ln-stat">
      {/* The animated value is decorative motion over text that is already
          correct in the DOM, so it is announced once rather than on every
          frame of the count. */}
      <span className="ln-stat-num" aria-live="off">
        {shown}
        {f.unit && <span className="ln-stat-unit">{f.unit}</span>}
      </span>
      <span className="ln-stat-label">{f.label}</span>
      <span className="ln-stat-note">{f.note}</span>
    </li>
  );
}

export function StatFigures() {
  const ref = useRef<HTMLUListElement>(null);
  // `amount: 0.2` so the count starts as the strip enters rather than after
  // it is fully past the fold. `once` — re-counting on every scroll-past is a
  // distraction, not a delight.
  const seen = useInView(ref, { once: true, amount: 0.2 });

  return (
    <ul ref={ref} className="ln-grid ln-stat-list">
      {FIGURES.map((f) => (
        <Figure key={f.label} f={f} run={seen} />
      ))}
    </ul>
  );
}
