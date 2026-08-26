/**
 * The gate ledger — the hero's one animated element.
 *
 * The generated UX rules cap this page at "1-2 key elements animated per view,
 * High severity", so motion is spent where the argument is rather than spread
 * across the page. This is that place: every code-memory tool can show a graph
 * going up; none of them can show their own improvement being *refused*. The
 * entries reveal in order so the refusal is the thing you land on.
 *
 * Both entries are real shapes from `EvalReport::is_committable()` — a rule
 * that held its canaries and one that did not.
 */
import { useLayoutEffect, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";

/** Generated rule `duration-timing`: 150-300ms for micro-interactions, and
 *  ease-out for entering. */
const EASE_OUT = [0.22, 1, 0.36, 1] as const;

const entries = [
  {
    verdict: "COMMITTED",
    ok: true,
    delta: "gen 4 · +6pp held-out",
    rule: "-exclude:tests/fixtures/* for class “migration”",
    why: "Fixtures ranked top-5 on 11 migration tasks and helped in none. Canaries held at 0.82.",
  },
  {
    verdict: "REFUSED",
    ok: false,
    delta: "canary 0.82 → 0.71",
    rule: "-exclude:src/db/pool.rs for class “timeout”",
    why: "Looked dead on the training split. Re-running the canaries showed it was the only answer to three of them. Not shipped.",
  },
];

export function LedgerPanel() {
  const still = useReducedMotion();

  // The entrance is applied only *after* mount, so `initial` never reaches the
  // server-rendered HTML. framer-motion writes its initial variant out as
  // inline `opacity: 0`, which means a hidden-by-default entrance ships a
  // blank panel to anyone whose JS is slow, disabled, or whose browser paused
  // rAF in a background tab — and this panel is the page's entire argument.
  // `useLayoutEffect` flips it before paint, so the animation still plays.
  const [armed, setArmed] = useState(false);
  useLayoutEffect(() => setArmed(true), []);

  return (
    <motion.aside
      className="ln-ledger"
      aria-label="Example policy gate ledger"
      initial={armed && !still ? "hidden" : false}
      animate="shown"
      variants={{
        hidden: {},
        shown: { transition: { staggerChildren: 0.14, delayChildren: 0.15 } },
      }}
    >
      <div className="ln-ledger-head">
        <span>policy ledger</span>
        <span>is_committable()</span>
      </div>

      {entries.map((e) => (
        <motion.div
          key={e.verdict}
          className="ln-entry"
          variants={{
            hidden: { opacity: 0, y: 8 },
            shown: {
              opacity: 1,
              y: 0,
              transition: { duration: 0.28, ease: EASE_OUT },
            },
          }}
        >
          <span className="ln-entry-delta">{e.delta}</span>
          <span
            className={`ln-verdict ${e.ok ? "ln-verdict-ok" : "ln-verdict-no"}`}
          >
            {e.verdict}
          </span>
          <p className="ln-entry-rule">{e.rule}</p>
          <p className="ln-entry-why">{e.why}</p>
        </motion.div>
      ))}
    </motion.aside>
  );
}
