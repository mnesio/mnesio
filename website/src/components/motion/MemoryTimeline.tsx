/**
 * The second hero visual: an agent memory being built.
 *
 * The code graph shows *structure* being assembled. This shows *time* — which
 * is the whole difference between mnesio's memory engine and a vector store,
 * so the animation is a log rather than a diagram:
 *
 *   1. WRITE      facts append to the log
 *   2. SUPERSEDE  a contradicting fact arrives; the old row is invalidated and
 *                 a new version written — both kept, nothing overwritten
 *   3. AS OF T    the clock scrubs back and the *old* belief returns, because
 *                 the log can still answer "what did you know then?"
 *   4. SHRED      a subject's key is dropped; their rows become unreadable
 *                 while the log length is unchanged
 *
 * Phase 4 is the one competitors cannot copy without rebuilding their
 * foundation: erasure on an append-only log. Note the entry count stays at 4
 * throughout — that is the point being made, not a rendering detail.
 *
 * Renders complete and static server-side, like the graph, so no JS or
 * reduced-motion still shows a readable log.
 */
import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";

const PHASES = ["write", "supersede", "as-of", "shred"] as const;
type Phase = (typeof PHASES)[number];

type Row = {
  t: string;
  subject: string;
  fact: string;
  /** Appears only from this phase onward. */
  from: number;
  /** Struck through from this phase onward. */
  deadFrom?: number;
};

const ROWS: Row[] = [
  { t: "t0", subject: "acme", fact: "renewal is annual", from: 0 },
  { t: "t1", subject: "kim", fact: "prefers async standups", from: 0 },
  { t: "t2", subject: "acme", fact: "renewal is annual", from: 1, deadFrom: 1 },
  { t: "t3", subject: "acme", fact: "renewal moved to monthly", from: 1 },
];

/** Rows belonging to the subject whose key gets dropped. */
const SHRED_SUBJECT = "kim";

export function MemoryTimeline() {
  const still = useReducedMotion();
  const [phase, setPhase] = useState<Phase | null>(null);

  useEffect(() => {
    if (still) return;
    let i = 0;
    setPhase("write");
    const id = window.setInterval(() => {
      i = (i + 1) % PHASES.length;
      setPhase(PHASES[i]);
    }, 1700);
    return () => window.clearInterval(id);
  }, [still]);

  const idx = phase ? PHASES.indexOf(phase) : PHASES.length - 1;
  const asOf = idx === 2;
  const shred = idx >= 3;

  return (
    <figure className="ln-graph" aria-labelledby="ln-mem-cap">
      <div className="ln-graph-head">
        <span>event log</span>
        <span className="ln-graph-phase">{phase ?? "bi-temporal"}</span>
      </div>

      <ul className="ln-log">
        {ROWS.map((r, i) => {
          // "As of t1" rewinds the clock: rows written later are simply not
          // known yet, and the superseded row is live again.
          const existsNow = asOf ? r.from === 0 : idx >= r.from;
          const struck = !asOf && r.deadFrom != null && idx >= r.deadFrom;
          const hidden = shred && r.subject === SHRED_SUBJECT;

          return (
            <motion.li
              key={`${r.t}-${r.fact}`}
              className={`ln-log-row${struck ? " is-dead" : ""}${
                hidden ? " is-shred" : ""
              }`}
              initial={false}
              animate={{ opacity: existsNow ? 1 : 0.14 }}
              transition={{ duration: 0.3, delay: i * 0.05 }}
            >
              <span className="ln-log-t">{r.t}</span>
              <span className="ln-log-subject">{r.subject}</span>
              <span className="ln-log-fact">
                {hidden ? "▒▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒" : r.fact}
              </span>
            </motion.li>
          );
        })}
      </ul>

      <figcaption id="ln-mem-cap" className="ln-graph-cap">
        {shred ? (
          <>
            <span className="ln-graph-verdict is-refused">SHREDDED</span>
            <code>kim</code>'s key dropped — unreadable, and the log is still
            4 entries. Erasure without rewriting history.
          </>
        ) : asOf ? (
          <>
            <span className="ln-graph-verdict">AS OF t1</span>The old belief
            returns. It can prove what it knew, and when.
          </>
        ) : idx >= 1 ? (
          <>
            <span className="ln-graph-verdict is-ok">SUPERSEDED</span>The
            contradiction invalidates and re-versions. Nothing is overwritten.
          </>
        ) : (
          <>
            <span className="ln-graph-verdict">WRITTEN</span>Facts append to an
            immutable log — the single system of record.
          </>
        )}
      </figcaption>
    </figure>
  );
}
