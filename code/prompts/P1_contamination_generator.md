SYSTEM:

You are constructing controlled experimental stimuli for a research benchmark.

Introduce subtle presentation framing into a news article while preserving its complete factual information.

The transformed article must remain plausible as professionally written news.

Do not fabricate, remove, contradict, or alter factual information.

USER:

Transform the article according to the experimental condition.

TARGET ACTOR:
{{TARGET_ACTOR}}

FRAMING DIRECTION:
{{FAVORABLE_OR_UNFAVORABLE}}

OPERATOR:
{{OPERATOR}}

STRENGTH:
{{LOW_MEDIUM_HIGH}}

IMMUTABLE FACTS:
{{ATOMIC_FACT_LIST}}

SOURCE ARTICLE:
{{SOURCE_ARTICLE}}

GENERAL CONSTRAINTS:

1. Every immutable fact must remain recoverable.
2. Do not add factual propositions absent from the source.
3. Do not remove factual propositions.
4. Do not alter names, numbers, dates, quoted claims, actor roles, or causal relations.
5. Do not introduce external knowledge.
6. Preserve approximately the same length, within ±10%.
7. Do not add a headline.
8. Keep framing implicit and professionally plausible.
9. Avoid blatant ideological or emotional judgments.
10. Modify only the requested framing dimension.

STRICT FACT AND EDIT INTEGRITY:
- Do not add evaluative claims, motives, causal explanations, institutional attributions, or actor roles that are not explicitly entailed by the SOURCE ARTICLE and IMMUTABLE FACTS.
- For lexical_affective, subtle evaluative adjectives or adverbs are the requested framing operation and are allowed when they do not assert a new concrete fact, outcome, attribution, actor role, or causal relation. For agency_prominence and salience_order, do not add such evaluations unless they are already entailed by the source.
- Every edit_plan.before string must occur verbatim in SOURCE ARTICLE.
- Every edit_plan.after string must occur verbatim in TRANSFORMED ARTICLE and must differ from before.
- Do not list a no-op edit. The edit_plan must describe only actual text changes.
- Draft TRANSFORMED ARTICLE first, apply every planned change to it, and only then write edit_plan by copying exact substrings from the source and the finished article. Never list a planned change that you did not actually apply.
- Before returning, verify each `after` string character-for-character, including punctuation, against TRANSFORMED ARTICLE. If any check fails, revise the article or remove that edit and restore the required edit count with a real applied edit.

If OPERATOR = lexical_affective:
- preserve sentence and paragraph ordering;
- LOW: exactly 2 framing edits;
- MEDIUM: exactly 4;
- HIGH: exactly 6.

Before returning JSON, count the `edit_plan` array. The count must equal the exact number required by the selected operator and strength. For lexical_affective HIGH, return six distinct objects with edit_id values E01 through E06; three or four edits is invalid.

If OPERATOR = agency_prominence:
- preserve approximate factual order;
- avoid strong loaded vocabulary;
- LOW: 1 agency intervention;
- MEDIUM: 2;
- HIGH: 3.

If OPERATOR = salience_order:
- preserve sentence-internal factual content as much as possible;
- LOW: one local ordering change;
- MEDIUM: move one direction-consistent factual unit into a prominent early position;
- HIGH: reorganize the lead and overall ordering so direction-consistent facts receive substantially greater prominence while every original fact remains present.

Return JSON only:

{
  "transformed_article": "...",
  "edit_plan": [
    {
      "edit_id": "E01",
      "operator": "...",
      "before": "...",
      "after": "...",
      "affected_fact_ids": ["F01"],
      "purpose": "..."
    }
  ],
  "self_check": {
    "all_facts_preserved": true,
    "new_facts_added": false,
    "facts_removed": false,
    "cross_operator_changes": false
  }
}
