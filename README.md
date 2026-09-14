# Recognizing Is Not Reversing

Experiment code and data for **Recognizing Is Not Reversing: A Controlled
Inversion Test of Fact-Preserving News Framing**.

Yi Liu · University of Science and Technology of China  
scnuliuyi@mail.ustc.edu.cn

[GitHub repository](https://github.com/daoisdao/Recognizing-Is-Not-Reversing-A-Controlled-Inversion-Test-of-Fact-Preserving-News-Framing)

```text
code/   Data construction, model experiments, prompts, configurations, and metrics
data/   News sources, canonical articles, facts, framed variants, and model results
```

## Analyze the included results

Python 3.10 or newer:

```sh
python -m pip install -r code/requirements.txt
python -B code/analyze_results.py --output data/analysis/results
```

This runs offline and produces overall, operator, strength, conditional-IRR,
and thinking-comparison results. Confidence intervals use 5,000 source-clustered
bootstrap replicates (seed 20260907). Choose a new output directory for each run.
The thinking analysis includes both available-output and strictly paired
comparisons; pairing uses the same sample IDs separately for D0 and R0.

## Experiment

The final dataset contains 60 source articles and 540 framed variants:
three operators × three strengths per source. Direction is fixed across
strengths for each source/operator pair, with 90 favorable and 90 unfavorable
assignments. The disjoint 10-source development set is excluded from final
statistics.

GLM-5.2 (`glm-5-2-260617`) generates and validates framing interventions.
Qwen (`qwen-plus`), DeepSeek (`deepseek-v4-flash`), and Kimi (`kimi-k2.6`)
perform blind detection (D0), self-guided reconstruction (R0), clean
reconstruction, and high-strength thinking experiments. R0 receives the article
and the model's own D0 result; the source, fact inventory, and edit plan are hidden.

FactF1 measures deterministic lexical/numeric compatibility with the source fact
inventory. IRR divides reversed edits by injected edits using normalized
before/after phrase matching and fuzzy token windows (threshold 0.88).
Before-only matches count as reversed; after-only as retained; both/neither
as ambiguous. Missing provider outputs remain missing observations.

See [code/README.md](code/README.md) for experiment commands and
[data/README.md](data/README.md) for dataset contents.

## Licenses

Code: [Apache-2.0](LICENSE).
Project-authored data adaptations and annotations: [CC BY 4.0](data/LICENSE.md),
with original news licensing and [source attribution](data/sources_attribution.json).
