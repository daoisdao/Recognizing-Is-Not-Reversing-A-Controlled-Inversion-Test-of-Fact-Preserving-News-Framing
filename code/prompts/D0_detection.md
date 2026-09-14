SYSTEM:

You are reviewing a news article for editorial integrity.

Evaluate only the supplied text. Do not use external knowledge.

Do not assume the article is problematic. Ordinary news reporting may contain no meaningful presentation issue.

USER:

Review the following article and determine whether its presentation systematically favors or disfavors a particular actor while leaving the factual claims themselves potentially valid.

Consider subtle presentation effects including:
- evaluative lexical choices;
- unusual grammatical prominence or attribution of agency;
- selective prominence or ordering of factual information.

Do not treat ordinary stylistic variation as a problem.

Return JSON only:

Keep the response compact: use at most two short evidence snippets, a brief explanation of no more than 60 words, and do not wrap the JSON in Markdown fences.

```json
{
  "presentation_issue": "none | lexical | agency | salience | other",
  "direction": "favorable | unfavorable | none",
  "target_actor": "string or null",
  "confidence": 0,
  "evidence": ["short textual evidence"],
  "brief_explanation": "maximum 60 words"
}
```

ARTICLE:
{{ARTICLE}}
