# Deterministic MCP Guardrails for Safe Power-Pathway Design Agents

Constraint-based benchmark and deterministic Model Context Protocol (MCP) guardrail testbed for grid-constrained power-pathway planning agents.

> CMU 17-745 Machine Learning in Production · Spring 2026 (Course Research Paper)
> Author: [Chenrui (Eric) Xu](https://chenrui-xu.github.io/) · Carnegie Mellon University
> Course Paper: [PDF](paper/mcp-guardrails-course-paper-2026.pdf) · Related Prototype: [power-pathway-optimizer](https://github.com/Chenrui-Xu/power-pathway-optimizer)

---

## Overview

LLM agents advising on data-center and industrial power-supply strategies frequently commit arithmetic errors or omit hard engineering constraints (firm dispatchable capacity, capital budget ceilings, carbon intensity limits, and system reliability targets).

This repository implements:
1. **PowerPathway-22 Benchmark (`src/benchmark/`)** — 22 test cases spanning five failure modes: infeasible phases, budget violations, carbon violations, hallucination traps, and feasible baselines.
2. **Deterministic Validators & MCP Gateway (`src/guardrails/`)** — deterministic physical/financial/carbon/reliability validators plus a stateful `MCPGateway` that blocks final recommendation submission until prerequisite validators have been invoked.
3. **Four Progressive Agent Conditions (`src/agent/`)**:
   - `B1`: Prompt-only baseline
   - `B2`: Prompt + self-critique
   - `B3`: Validator tools available without gateway enforcement
   - `G1`: MCP-guarded agent with mandatory pre-execution validator sequencing
4. **Evaluation Harness (`src/evaluation/`)** — metrics computation, offline mock runner, and live Vertex AI runner (`gemini-2.5-flash`).

### Summary of Reported Paper Results (88 Live Runs, Gemini 2.5 Flash)

| Condition | Description | Invalid-Decision Rate |
|---|---|---|
| `B1` | Prompt-only | 36.4% (8/22) |
| `B2` | Self-critique | 36.4% (8/22) |
| `B3` | Tools without enforcement | 9.1% (2/22) |
| `G1` | MCP-guarded (mandatory validators) | 4.5% (1/22) |

Raw run records from the paper are archived under `data/results/` (`run_merged_final.json` for `B1`/`B2`/`G1` and `run_b3_full.json` for `B3`).

---

## Quick Start

### 1. Install and Run Unit Tests (No API Key Required)

```bash
git clone https://github.com/Chenrui-Xu/mcp-guardrails-power-pathway.git
cd mcp-guardrails-power-pathway
pip install -e ".[dev]"
pytest
```

Expected: **81 tests passing** covering benchmark cases, deterministic validators, MCP gateway sequencing enforcement, and evaluation metrics.

### 2. Offline Synthetic Evaluation (No API Key Required)

```bash
PYTHONPATH=src python -m evaluation.mock_runner
```

### 3. Live Evaluation via Google Vertex AI

Install the optional Vertex AI dependency and configure your GCP project via environment variables:

```bash
pip install -e ".[vertex]"
export GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
export GOOGLE_CLOUD_LOCATION=us-central1
PYTHONPATH=src python -m evaluation.runner --agents B1 B2 B3 G1
```

---

## Repository Structure

```text
mcp-guardrails-power-pathway/
├── src/
│   ├── benchmark/
│   │   ├── tech_params.py     # Reference technology specs (capex, carbon, CF, deploy time)
│   │   └── cases.py           # PowerPathway-22 benchmark cases & ground-truth feasibility
│   ├── guardrails/
│   │   ├── tools.py           # Deterministic capacity, cost, carbon & reliability validators
│   │   └── mcp_gateway.py     # Stateful MCP enforcement gateway
│   ├── agent/
│   │   ├── base_agent.py      # Shared prompt, JSON parser & Vertex AI config
│   │   ├── b1_agent.py        # Prompt-only baseline
│   │   ├── b2_agent.py        # Self-critique baseline
│   │   ├── b3_agent.py        # Unguarded tool-calling agent
│   │   └── g1_agent.py        # MCP-guarded agent
│   └── evaluation/
│       ├── metrics.py         # Invalid-decision & constraint-violation metrics
│       ├── mock_runner.py     # Offline synthetic runner
│       └── runner.py          # Live evaluation runner
├── data/
│   ├── cases/benchmark_cases.json
│   └── results/               # Archived 88-run paper results (B1/B2/B3/G1)
├── tests/                     # 81 unit & gateway enforcement tests
└── paper/                     # Course paper LaTeX source & PDF
```

---

## Citation

```bibtex
@misc{xu2026mcpguardrails,
  author       = {Xu, Chenrui},
  title        = {Deterministic MCP Guardrails for Safe Power-Pathway Design Agents},
  year         = {2026},
  howpublished = {Carnegie Mellon University, 17-745 Machine Learning in Production (Spring 2026)},
  url          = {https://github.com/Chenrui-Xu/mcp-guardrails-power-pathway}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
