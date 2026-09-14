SYSTEM:

You are independently validating controlled experimental news stimuli. Judge the transformed article strictly against the source and immutable fact inventory.

USER:

SOURCE:
{{SOURCE_ARTICLE}}

TRANSFORMED:
{{TRANSFORMED_ARTICLE}}

IMMUTABLE FACTS:
{{ATOMIC_FACTS}}

REQUESTED OPERATOR:
{{OPERATOR}}

REQUESTED DIRECTION:
{{DIRECTION}}

REQUESTED STRENGTH:
{{STRENGTH}}

EDIT PLAN:
{{EDIT_PLAN}}

Check:
1. every immutable fact remains entailed;
2. no unsupported factual claim was added;
3. no entity, quantity, date, attribution, actor role, or causal relation changed;
4. requested operator is present;
5. requested direction is present;
6. framing remains subtle and professionally plausible;
7. there is no substantial cross-operator manipulation.

Reject the sample if the transformed article adds a concrete factual claim, motive, causal explanation, institutional attribution, actor role, quantity, date, or outcome not entailed by the source or immutable facts. For lexical_affective, subtle evaluative adjectives and adverbs are the requested manipulation; do not mark them unsupported solely because they express positive or negative tone. Still reject them if they assert a new concrete outcome, causal relation, institutional attribution, or actor role. For agency_prominence and salience_order, reject added evaluations unless they are entailed by the source. Independently verify every edit-plan entry: before must be an exact substring of SOURCE, after must be an exact substring of TRANSFORMED, and before and after must not be identical. Put concrete unsupported additions in unsupported_new_facts and concrete changes in altered_facts.

Return JSON only:

{
  "pass": true,
  "fact_preservation": {
    "F01": true
  },
  "unsupported_new_facts": [],
  "altered_facts": [],
  "operator_valid": true,
  "direction_valid": true,
  "strength_plausible": true,
  "professionally_plausible": true,
  "cross_operator_leakage": false,
  "failure_reason": null
}

A sample passes only if all immutable facts are preserved and no unsupported factual proposition has been introduced.
