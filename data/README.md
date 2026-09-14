# Experiment data

- `raw/`: 60 news source texts and provenance.
- `canonical/`: 60 pre-intervention reference articles.
- `facts/`: atomic fact inventories.
- `contaminated/`: 540 framed articles, edit plans, and validation records.
- `splits/`: final source IDs, condition assignments, and selection records.
- `development/`: the disjoint 10-source ID/title/provenance inventory,
  excluded from final statistics.
- `results/evaluated/`: Qwen, DeepSeek, and Kimi D0/R0 outputs in direct
  and thinking modes, plus direct clean reconstruction controls.
- `results/metrics/`: per-record FactF1 and IRR.
- `sources_attribution.json`: Wikinews titles, attribution, and revision links.

Provider-blocked and execution-missing outputs are recorded as missing observations.
Some historical `publication_date` fields derive from source revision timestamps;
revision links provide the associated source history.

Project-authored adaptations and annotations use [CC BY 4.0](LICENSE.md).
Original news text retains its upstream licenses and attribution.
