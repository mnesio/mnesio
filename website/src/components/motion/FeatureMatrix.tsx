/**
 * Feature-by-feature comparison, one matrix per engine.
 *
 * The engine panels above carry a three-column summary, which is enough to
 * place each product but not enough to actually compare it. This is the
 * breakdown: every dimension that separates these systems, scored for each,
 * with the losing rows kept in.
 *
 * Two rules this follows, because they are the difference between a comparison
 * and an advertisement:
 *
 *   1. Every row where a competitor leads is marked as a win *for them*, in
 *      the same visual language as our own wins. Six of the rows here are.
 *   2. Every mnesio cell states the measurement, not an adjective — "37.7 min,
 *      73,567 symbols" rather than "fast".
 *
 * Sources: `COMPETITORS.md`, `crates/mnesio-bench/manifest/` and the published
 * work each competitor links from its own README.
 */
import { useState } from "react";
import { motion, useReducedMotion } from "framer-motion";

type Verdict = "us" | "them" | "tie" | "none";
type Cell = { v: Verdict; t: string };
type Row = { feature: string; cells: Cell[] };
type Matrix = { id: string; label: string; systems: string[]; rows: Row[] };

const CODE: Matrix = {
  id: "code",
  label: "Code memory",
  systems: [
    "mnesio",
    "graphify",
    "codebase-memory-mcp",
    "Cursor · Sourcegraph",
  ],
  rows: [
    {
      feature: "Symbol-level graph",
      cells: [
        { v: "tie", t: "symbols + call edges" },
        { v: "tie", t: "yes" },
        { v: "tie", t: "yes" },
        { v: "tie", t: "yes" },
      ],
    },
    {
      feature: "Language grammars",
      cells: [
        { v: "none", t: "30" },
        { v: "none", t: "36" },
        { v: "them", t: "158, vendored" },
        { v: "none", t: "many" },
      ],
    },
    {
      feature: "Type resolution",
      cells: [
        { v: "none", t: "name-match — 23–46% of edges" },
        { v: "none", t: "none stated" },
        { v: "them", t: "hybrid LSP, 12 languages" },
        { v: "them", t: "LSP-grade" },
      ],
    },
    {
      feature: "Ranking policy",
      cells: [
        { v: "us", t: "3 signals, each ablated" },
        { v: "none", t: "unspecified" },
        { v: "tie", t: "11 signals, enumerated" },
        { v: "none", t: "proprietary" },
      ],
    },
    {
      feature: "Index persistence",
      cells: [
        { v: "tie", t: "warm restart 9.6× — byte-identical NOT met" },
        { v: "none", t: "—" },
        { v: "them", t: "SQLite, survives restart" },
        { v: "tie", t: "server-side" },
      ],
    },
    {
      feature: "Scale proof",
      cells: [
        { v: "none", t: "5.35M LOC, 73,567 symbols, 37.7 min" },
        { v: "none", t: "—" },
        { v: "them", t: "28M LOC, 75k files, 3 min" },
        { v: "them", t: "whole-org indexes" },
      ],
    },
    {
      feature: "Published eval",
      cells: [
        { v: "none", t: "400 tasks, 4 repos — not yet published" },
        { v: "none", t: "—" },
        { v: "them", t: "31 repos, arXiv paper" },
        { v: "none", t: "—" },
      ],
    },
    {
      feature: "Records edit outcomes",
      cells: [
        { v: "tie", t: "build, tests, accept/reject" },
        { v: "tie", t: "useful / dead_end / corrected" },
        { v: "none", t: "no" },
        { v: "none", t: "no" },
      ],
    },
    {
      feature: "Gated improvement",
      cells: [
        { v: "us", t: "canaries re-run; a regression is refused" },
        { v: "none", t: "writes LESSONS.md — nothing can fail" },
        { v: "none", t: "no" },
        { v: "none", t: "no" },
      ],
    },
  ],
};

const MEMORY: Matrix = {
  id: "memory",
  label: "Agent memory",
  systems: ["mnesio", "Zep / Graphiti", "Mem0", "Letta · Cognee · LangMem"],
  rows: [
    {
      feature: "Semantic recall",
      cells: [
        { v: "tie", t: "hybrid vector + BM25" },
        { v: "tie", t: "yes" },
        { v: "tie", t: "yes" },
        { v: "tie", t: "yes" },
      ],
    },
    {
      feature: "Append-only + bi-temporal",
      cells: [
        { v: "us", t: "invalidate-and-supersede, never overwrite" },
        { v: "tie", t: "yes" },
        { v: "none", t: "no" },
        { v: "none", t: "no" },
      ],
    },
    {
      feature: "Time-travel to any past T",
      cells: [
        { v: "us", t: "snapshot_as_of(T) reconstructs the live set" },
        { v: "none", t: "partial" },
        { v: "none", t: "partial" },
        { v: "none", t: "no" },
      ],
    },
    {
      feature: "Provenance for a belief",
      cells: [
        { v: "us", t: "every fact traced to its source events" },
        { v: "none", t: "partial" },
        { v: "none", t: "partial" },
        { v: "none", t: "no" },
      ],
    },
    {
      feature: "Erasure",
      cells: [
        { v: "us", t: "crypto-shred — drop the subject key" },
        { v: "none", t: "no" },
        { v: "none", t: "no" },
        { v: "none", t: "delete row, history lost" },
      ],
    },
    {
      feature: "Self-improvement",
      cells: [
        { v: "us", t: "gated — refused if canaries regress" },
        { v: "none", t: "no" },
        { v: "none", t: "no" },
        { v: "none", t: "self-edit, ungated" },
      ],
    },
    {
      feature: "KV cache as a gated view",
      cells: [
        { v: "us", t: "versioned, erasable cartridges" },
        { v: "none", t: "no" },
        { v: "none", t: "no" },
        { v: "none", t: "no" },
      ],
    },
    {
      feature: "Published LoCoMo / LongMemEval",
      cells: [
        { v: "none", t: "not yet" },
        { v: "them", t: "63.8% LongMemEval" },
        { v: "them", t: "92.5% LoCoMo, self-reported" },
        { v: "them", t: "Hindsight 94.6% LongMemEval" },
      ],
    },
  ],
};

const MATRICES = [CODE, MEMORY];

export function FeatureMatrix() {
  const [active, setActive] = useState(0);
  const still = useReducedMotion();
  const m = MATRICES[active];

  const wins = m.rows.filter((r) => r.cells[0].v === "us").length;
  const losses = m.rows.filter((r) =>
    r.cells.slice(1).some((c) => c.v === "them"),
  ).length;
  const rows = (n: number) => `${n} row${n === 1 ? "" : "s"}`;

  return (
    <div className="ln-matrix">
      <div className="ln-matrix-tabs" role="tablist" aria-label="Engine">
        {MATRICES.map((x, i) => (
          <button
            key={x.id}
            role="tab"
            aria-selected={i === active}
            className={`ln-matrix-tab${i === active ? " is-active" : ""}`}
            onClick={() => setActive(i)}
          >
            {x.label}
          </button>
        ))}
        <p className="ln-matrix-score">
          {rows(wins)} only mnesio fills · {rows(losses)} a competitor leads
        </p>
      </div>

      {/*
        No `AnimatePresence mode="wait"` here, deliberately.

        With it, the incoming table does not mount until the outgoing one has
        finished exiting — and a browser pauses framer-motion's frame loop in a
        background tab, so that exit never completed: the panel sat showing the
        *code* matrix while both the tab and the score line read "Agent
        memory". Switching tabs is a content change, and content must never
        wait on an animation to finish.

        Keying on `m.id` re-mounts the table the instant `active` changes; the
        entrance is decoration layered over a swap that already happened.
      */}
      <motion.div
        key={m.id}
        className="ln-matrix-scroll"
        initial={still ? false : { opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
      >
        <table className="ln-matrix-table">
          <thead>
            <tr>
              <th scope="col">Capability</th>
              {m.systems.map((sys, i) => (
                <th key={sys} scope="col" className={i === 0 ? "is-us" : ""}>
                  {sys}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {m.rows.map((r) => (
              <tr key={r.feature}>
                <th scope="row">{r.feature}</th>
                {r.cells.map((c, i) => (
                  <td key={i} className={`v-${c.v}${i === 0 ? " is-us" : ""}`}>
                    {c.t}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </motion.div>

      <p className="ln-matrix-note">
        Green marks whoever leads a row, including when that is not us. Every
        mnesio cell states the measurement rather than an adjective; the numbers
        are reproducible from <code>crates/mnesio-bench/manifest/</code>.
      </p>
    </div>
  );
}
