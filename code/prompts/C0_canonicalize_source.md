SYSTEM:

You are constructing a controlled research benchmark on news representation. Convert the supplied licensed news article into a neutral canonical source. Use only information in the article. Do not add, infer, update, or correct facts using external knowledge.

USER:

Create a canonical English news article of 180–260 words.

Preserve all key factual propositions, named entities, numbers, dates, attribution, actor roles, and causal claims. Remove source lists, image captions, navigation, repetition, and irrelevant stylistic clutter. Avoid strong evaluative language. Do not add a headline.

Return JSON only:

{
  "canonical_text": "...",
  "target_actor": "...",
  "self_check": {
    "source_only": true,
    "key_facts_preserved": true,
    "entities_numbers_dates_preserved": true,
    "neutral_wording": true
  }
}

REQUIRED TARGET ACTOR:
{{TARGET_ACTOR}}

SOURCE ARTICLE:
{{SOURCE_ARTICLE}}
