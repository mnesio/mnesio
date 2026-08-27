/**
 * The hero visual: a code graph being built, queried, and gated.
 *
 * This is the product's actual loop rather than decoration, so it runs as four
 * phases on a cycle:
 *
 *   1. PARSE   symbols appear as files are read
 *   2. LINK    call edges draw between them as names resolve
 *   3. RETRIEVE a task lights the symbols packed into context
 *   4. GATE    one of them is refused, and the edge to it is struck out
 *
 * Phase 4 is the point. Every code tool can show a graph assembling; the thing
 * only mnesio can show is its own suggestion being *rejected*, so the cycle
 * ends there and holds a beat before repeating.
 *
 * Node names are real symbols from this repository — `pack`, `resolve`,
 * `Bm25View` and the rest are things you can grep for in `crates/`.
 *
 * Robustness: the graph renders complete and static on the server, and phases
 * only start after mount. A visitor with no JS, a paused frame loop, or
 * `prefers-reduced-motion` sees the finished graph rather than an empty box.
 */
import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";

type Seed = { id: string; label: string; tier: 0 | 1 | 2 };
type Node = Seed & { x: number; y: number; w: number };

/* Tiers are rows: a call graph reads top-down. */
const SEEDS: Seed[][] = [
  [{ id: "ctx", label: "code_context", tier: 0 }],
  [
    { id: "ret", label: "HybridRetriever", tier: 1 },
    { id: "bm", label: "Bm25View", tier: 1 },
    { id: "vec", label: "VectorView", tier: 1 },
  ],
  [
    { id: "res", label: "resolve", tier: 2 },
    { id: "pack", label: "pack", tier: 2 },
    { id: "graph", label: "CodeGraph", tier: 2 },
    { id: "fix", label: "fixtures", tier: 2 },
  ],
];

const VIEW_W = 460;
const NODE_H = 26;
const ROW_Y = [34, 116, 200];
/* JetBrains Mono at 10.5px advances ~6.3px per character. Boxes are sized from
   the label rather than fixed: a 15-character symbol like `HybridRetriever`
   overflowed an 84px box by 34px, which is what put text outside the cards. */
const CHAR_W = 6.3;
const PAD_X = 14;
const GAP = 14;

const nodeWidth = (label: string) =>
  Math.round(label.length * CHAR_W + PAD_X * 2);

/** Lay each tier out centred, with equal gaps, from the measured widths. */
const NODES: Node[] = SEEDS.flatMap((row, tier) => {
  const widths = row.map((n) => nodeWidth(n.label));
  const total = widths.reduce((a, b) => a + b, 0) + GAP * (row.length - 1);
  let cursor = (VIEW_W - total) / 2;
  return row.map((n, i) => {
    const w = widths[i];
    const x = cursor + w / 2;
    cursor += w + GAP;
    return { ...n, x, y: ROW_Y[tier], w };
  });
});

const EDGES: [string, string][] = [
  ["ctx", "ret"],
  ["ctx", "bm"],
  ["ctx", "vec"],
  ["ret", "res"],
  ["ret", "pack"],
  ["bm", "pack"],
  ["vec", "graph"],
  ["bm", "fix"],
];

/** Packed into context on the example task. */
const RETRIEVED = new Set(["ctx", "ret", "bm", "pack", "res"]);
/** The one the gate throws out — fixtures ranked high and never helped. */
const REFUSED = "fix";

const PHASES = ["parse", "link", "retrieve", "gate"] as const;
type Phase = (typeof PHASES)[number];

/** Anchor points on the node box edges, so edges meet boxes rather than centres. */
function anchor(n: Node, side: "top" | "bottom") {
  return { x: n.x, y: side === "top" ? n.y - NODE_H / 2 : n.y + NODE_H / 2 };
}

function edgePath(a: Node, b: Node) {
  const from = anchor(a, "bottom");
  const to = anchor(b, "top");
  const mid = (from.y + to.y) / 2;
  // Orthogonal-ish routing with a soft knee reads as a call graph rather than
  // a spider web.
  return `M ${from.x} ${from.y} C ${from.x} ${mid}, ${to.x} ${mid}, ${to.x} ${to.y}`;
}

const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));

export function CodeGraph() {
  const still = useReducedMotion();
  // `null` means "not running": the finished graph, which is what the server
  // renders and what a reduced-motion visitor keeps.
  const [phase, setPhase] = useState<Phase | null>(null);

  useEffect(() => {
    if (still) return;
    let i = 0;
    setPhase("parse");
    const id = window.setInterval(() => {
      i = (i + 1) % PHASES.length;
      setPhase(PHASES[i]);
      // The gate verdict holds longer than the other phases — it is the one
      // frame a reader should actually stop on.
    }, 1700);
    return () => window.clearInterval(id);
  }, [still]);

  const idx = phase ? PHASES.indexOf(phase) : PHASES.length - 1;
  const shownNodes = phase === "parse" ? 1 : 1; // nodes stagger via delay below
  const linked = idx >= 1;
  const lit = idx >= 2;
  const gated = idx >= 3;

  return (
    <figure className="ln-graph" aria-labelledby="ln-graph-cap">
      <div className="ln-graph-head">
        <span>code graph</span>
        <span className="ln-graph-phase">{phase ?? "indexed"}</span>
      </div>

      <svg viewBox="0 0 460 250" role="img" aria-hidden="true">
        {/* --- edges --- */}
        <g fill="none" strokeWidth="1.25">
          {EDGES.map(([a, b], i) => {
            const isRefusedEdge = b === REFUSED;
            const active = lit && RETRIEVED.has(a) && RETRIEVED.has(b);
            return (
              <motion.path
                key={`${a}-${b}`}
                d={edgePath(byId[a], byId[b])}
                stroke={
                  gated && isRefusedEdge
                    ? "var(--ln-refuse, #f87171)"
                    : active
                      ? "var(--ln-accent, #22c55e)"
                      : "var(--ln-line-strong, #3a4a68)"
                }
                initial={false}
                animate={{
                  pathLength: still ? 1 : linked ? 1 : 0,
                  opacity: gated && isRefusedEdge ? 0.5 : 1,
                }}
                transition={{
                  pathLength: { duration: 0.55, delay: linked ? i * 0.05 : 0 },
                  opacity: { duration: 0.25 },
                }}
              />
            );
          })}
        </g>

        {/* --- nodes --- */}
        {NODES.map((n, i) => {
          const isRefused = gated && n.id === REFUSED;
          const isLit = lit && RETRIEVED.has(n.id) && !isRefused;
          return (
            <motion.g
              key={n.id}
              initial={false}
              animate={{
                opacity: still ? 1 : phase === "parse" ? 0 : 1,
                scale: isRefused ? 0.94 : 1,
              }}
              transition={{
                opacity: {
                  duration: 0.3,
                  delay: phase === "parse" ? 0 : i * 0.055,
                },
                scale: { duration: 0.25 },
              }}
              style={{ transformOrigin: `${n.x}px ${n.y}px` }}
            >
              <rect
                x={n.x - n.w / 2}
                y={n.y - NODE_H / 2}
                width={n.w}
                height={NODE_H}
                /* Square, matching the rest of the page. */
                fill={isLit ? "rgba(34,197,94,0.16)" : "rgba(15,23,42,0.92)"}
                stroke={
                  isRefused
                    ? "var(--ln-refuse, #f87171)"
                    : isLit
                      ? "var(--ln-accent, #22c55e)"
                      : "var(--ln-line-strong, #3a4a68)"
                }
                strokeWidth="1.25"
              />
              <text
                x={n.x}
                y={n.y + 4}
                textAnchor="middle"
                className={`ln-graph-label${isLit ? " is-lit" : ""}${
                  isRefused ? " is-refused" : ""
                }`}
              >
                {n.label}
              </text>
              {/* Strike-through on the refused symbol: the gate did not just
                  rank it lower, it removed it from the context. */}
              {isRefused && (
                <motion.line
                  x1={n.x - n.w / 2 + 6}
                  y1={n.y}
                  x2={n.x + n.w / 2 - 6}
                  y2={n.y}
                  stroke="var(--ln-refuse, #f87171)"
                  strokeWidth="1.25"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.3 }}
                />
              )}
            </motion.g>
          );
        })}
      </svg>

      <figcaption id="ln-graph-cap" className="ln-graph-cap">
        {gated ? (
          <>
            <span className="ln-graph-verdict is-refused">REFUSED</span>
            <code>fixtures</code> ranked top-5 and helped in none — dropped
            before it reached the model.
          </>
        ) : lit ? (
          <>
            <span className="ln-graph-verdict is-ok">PACKED</span>5 symbols to a
            token budget, each tagged with why it was included.
          </>
        ) : linked ? (
          <>
            <span className="ln-graph-verdict">LINKED</span>Call edges resolved
            from the parse — the graph is rebuildable from the log.
          </>
        ) : (
          <>
            <span className="ln-graph-verdict">PARSED</span>A file is a source,
            a symbol is a memory. No new event types.
          </>
        )}
      </figcaption>
    </figure>
  );
}
