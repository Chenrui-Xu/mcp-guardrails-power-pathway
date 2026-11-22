"""
Experiment runner.

Usage:
  python src/evaluation/runner.py [--agents B1 B2 G1] [--cases F-001 IP-001 ...]
                                  [--output data/results/run_YYYYMMDD.json]
"""

from __future__ import annotations
import argparse
import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.cases import get_all_cases, get_case, PowerPathwayCase
from agent.b1_agent import run_b1
from agent.b2_agent import run_b2
from agent.b3_agent import run_b3
from agent.g1_agent import run_g1
from evaluation.metrics import evaluate_run, RunMetrics
from agent.base_agent import AgentRun


AGENT_RUNNERS = {"B1": run_b1, "B2": run_b2, "B3": run_b3, "G1": run_g1}


def run_experiment(
    agent_types: list[str],
    cases: list[PowerPathwayCase],
    output_path: Path,
    sleep_between: float = 1.0,
) -> list[dict]:
    results = []
    total = len(agent_types) * len(cases)
    done = 0

    for agent_type in agent_types:
        runner_fn = AGENT_RUNNERS[agent_type]
        for case in cases:
            done += 1
            print(f"[{done}/{total}] {agent_type} × {case.case_id} ({case.failure_mode})...",
                  end=" ", flush=True)
            run: AgentRun = runner_fn(case)
            metrics: RunMetrics = evaluate_run(run, case)
            result = {
                "case_id": case.case_id,
                "failure_mode": case.failure_mode,
                "agent_type": agent_type,
                "gt_feasible": case.gt_feasible,
                # Run outputs
                "final_recommendation": run.final_recommendation,
                "tool_calls": run.tool_calls,
                "blocked_calls": run.blocked_calls,
                "total_tokens": run.total_tokens,
                "latency_sec": run.latency_sec,
                "error": run.error,
                # Metrics
                "invalid_recommendation": metrics.invalid_recommendation,
                "correct_feasibility": metrics.correct_feasibility,
                "constraint_violations": metrics.constraint_violations,
                "validators_called": metrics.validators_called,
                "parse_failed": metrics.parse_failed,
            }
            status = "✓" if not metrics.invalid_recommendation else "✗"
            print(f"{status} invalid={metrics.invalid_recommendation} "
                  f"correct={metrics.correct_feasibility} "
                  f"tokens={run.total_tokens}")
            results.append(result)
            time.sleep(sleep_between)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results → {output_path}")
    return results


def print_summary(results: list[dict]) -> None:
    from collections import defaultdict
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_agent[r["agent_type"]].append(r)

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    header = f"{'Agent':<6} {'N':>4} {'InvalidRec%':>12} {'CorrectFeas%':>13} "
    header += f"{'AvgTokens':>10} {'AvgLatency':>11}"
    print(header)
    print("-"*70)

    for agent_type in ["B1", "B2", "B3", "G1"]:
        runs = by_agent.get(agent_type, [])
        if not runs:
            continue
        n = len(runs)
        invalid_pct = 100 * sum(r["invalid_recommendation"] for r in runs) / n
        correct = [r for r in runs if r["correct_feasibility"] is not None]
        correct_pct = 100 * sum(r["correct_feasibility"] for r in correct) / len(correct) if correct else 0
        avg_tok = sum(r["total_tokens"] for r in runs) / n
        avg_lat = sum(r["latency_sec"] for r in runs) / n
        print(f"{agent_type:<6} {n:>4} {invalid_pct:>11.1f}% {correct_pct:>12.1f}% "
              f"{avg_tok:>10.0f} {avg_lat:>10.1f}s")

    print("\nBy failure mode:")
    modes = sorted({r["failure_mode"] for r in results})
    for mode in modes:
        mode_results = [r for r in results if r["failure_mode"] == mode]
        print(f"\n  {mode}:")
        for agent_type in ["B1", "B2", "B3", "G1"]:
            runs = [r for r in mode_results if r["agent_type"] == agent_type]
            if not runs:
                continue
            inv = sum(r["invalid_recommendation"] for r in runs)
            print(f"    {agent_type}: {inv}/{len(runs)} invalid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", default=["B1", "B2", "B3", "G1"],
                        choices=["B1", "B2", "B3", "G1"])
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Case IDs to run (default: all)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    all_cases = get_all_cases()
    if args.cases:
        selected = [get_case(cid) for cid in args.cases]
    else:
        selected = all_cases

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else \
        Path(__file__).parent.parent.parent / f"data/results/run_{ts}.json"

    print(f"Running {args.agents} × {len(selected)} cases = "
          f"{len(args.agents) * len(selected)} total runs")

    results = run_experiment(args.agents, selected, output_path, args.sleep)
    print_summary(results)


if __name__ == "__main__":
    main()
