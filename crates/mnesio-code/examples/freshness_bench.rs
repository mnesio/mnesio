//! Time the freshness check against a real repository.
//!
//! ```sh
//! cargo run --release -p mnesio-code --features tree-sitter \
//!   --example freshness_bench -- <repo> [iterations]
//! ```
//!
//! The claim under test is that the no-change path is cheap enough to run on
//! *every* query, because that is what makes automatic freshness affordable.
//! An argument is not a measurement, so this measures it — and separately
//! measures the parse it deliberately avoids, since the gap between the two is
//! the entire point of splitting them.
//!
//! Run it `--release`. A debug build measures the wrong thing by a large
//! factor and would make the check look far worse than it is.

use std::time::Instant;

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let repo = args.first().cloned().unwrap_or_else(|| ".".into());
    let iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);

    // One warm pass: the first walk of a cold tree measures the OS page cache,
    // not our code, and an agent's second query is the case that matters.
    let _ = mnesio_code::memory::bench_fingerprint(&repo);

    let mut walk = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = mnesio_code::memory::bench_fingerprint(&repo);
        walk.push(t.elapsed().as_micros());
    }
    walk.sort_unstable();

    let t = Instant::now();
    let parsed = mnesio_code::memory::bench_parse(&repo);
    let parse_us = t.elapsed().as_micros();
    let (files, symbols) = parsed.unwrap_or((0, 0));

    println!("repo: {repo}");
    println!("{files} files, {symbols} symbols, {iters} iterations\n");
    println!("| stage | p50 | p95 | max |");
    println!("|---|---|---|---|");
    println!(
        "| freshness check (metadata only) | {:.2} ms | {:.2} ms | {:.2} ms |",
        percentile(&walk, 0.50) as f64 / 1000.0,
        percentile(&walk, 0.95) as f64 / 1000.0,
        walk.last().copied().unwrap_or(0) as f64 / 1000.0,
    );
    println!(
        "| full parse (only on change) | {:.2} ms | — | — |",
        parse_us as f64 / 1000.0
    );
    let ratio = parse_us as f64 / percentile(&walk, 0.50).max(1) as f64;
    println!("\nthe check avoids ~{ratio:.0}x the work when nothing changed.");
}
