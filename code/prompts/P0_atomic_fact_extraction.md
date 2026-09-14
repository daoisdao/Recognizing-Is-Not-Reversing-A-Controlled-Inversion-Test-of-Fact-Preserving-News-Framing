SYSTEM:

You are constructing a controlled research benchmark on news representation. Identify only atomic factual propositions explicitly supported by the article. Do not use external knowledge or infer unsupported facts.

USER:

Extract 6–10 atomic factual propositions from the following news article.

Each proposition must:
- express only one factual claim;
- preserve named entities, dates, quantities, actions, and attributed statements;
- avoid evaluative language;
- be independently verifiable from the article;
- preserve attribution.

Return JSON only:

{
  "facts": [
    {
      "fact_id": "F01",
      "proposition": "...",
      "entities": ["..."],
      "numbers_dates": ["..."]
    }
  ]
}

ARTICLE:
{{SOURCE_ARTICLE}}
