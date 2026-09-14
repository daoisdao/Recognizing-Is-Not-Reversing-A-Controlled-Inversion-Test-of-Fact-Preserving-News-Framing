# Experiment code

Run commands from the repository root. Install dependencies with:

```sh
python -m pip install -r code/requirements.txt
```

## Stages

- `scripts/00–02`: environment setup, source loading, and filtering.
- `scripts/03–06`: canonicalization, atomic facts, framing generation, and validation.
- `scripts/07–10`: D0, R0, clean reconstruction, and high-strength thinking.
- `scripts/11–12`: FactF1 and IRR.
- `scripts/13_analyze.py`: source-clustered analysis.
- `scripts/15_build_final_wikinews_sources.py`: final source-set construction.
- `scripts/16_analyze_reasoning_deltas.py`: historical available-output deltas.
- `scripts/repair_*.py`: canonical, fact, and contamination repairs used in construction.
- `analyze_results.py`: offline overall, conditional-IRR, and paired thinking analysis.
- `prompts/`: seven task templates used in the experiment.
- `configs/`: experimental settings and model adapters.

## Analyze existing results

```sh
python -B code/analyze_results.py --output data/analysis/results
```

No API requests are needed. Outputs are CSV and JSON files.

## Run model experiments

Set provider credentials as process environment variables; `.env.example`
lists the variables, model IDs, and endpoints. Provider inference is billed.

```sh
python -B code/scripts/00_validate_env.py --full --require-evaluated-api
python -B code/scripts/07_run_detection.py --full --family deepseek --workers 1
python -B code/scripts/08_run_reconstruction.py --full --family deepseek --workers 1
python -B code/scripts/09_run_clean_control.py --full --family deepseek --workers 1
python -B code/scripts/10_run_reasoning_ablation.py --full --family deepseek --workers 1
```

Use `qwen` or `kimi` for the other evaluated families. For a fresh experiment,
use a separate working copy and move its included
`data/results/evaluated/` and `data/results/metrics/` archives aside.
The `--resume` option continues an existing run.

Recompute reconstruction metrics:

```sh
python -B code/scripts/11_evaluate_fact_f1.py --full --family deepseek
python -B code/scripts/12_evaluate_irr.py --full --family deepseek
```

Add `--reasoning-mode thinking` for thinking outputs.

## Rebuild the dataset

Retain `data/raw/full_sources.jsonl` as input and move existing downstream
artifacts aside in a separate working copy. Stage 01 uses
`--input data/raw/full_sources.jsonl --resume`; stages 02–06 create
canonical articles, facts, framing variants, and validation records.

Six final variants used source-only repair before P2 acceptance:
S019 agency medium/high; S031, S034, and S042 agency high; S069 lexical high.
Generation and validation are separate calls to the same GLM model.

Use `--help` on each script for its arguments.
