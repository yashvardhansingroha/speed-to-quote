# Labelling guide for eval/eval_unlabeled.csv

Fill every empty label cell, then save the file as **eval/eval_labeled.csv** (same columns, same `eval_id` values).
Nothing in this project fills, suggests or checks labels against a "right answer": your judgement is the reference.
Read only the `message` column (and `received_date` for relative phrases like "next month").

Use exactly these values (lower case, underscores):

## intent  (one value per message)
`new_quote` · `renewal_or_replacement` · `coverage_question` · `document_or_info_provided` · `follow_up` · `existing_policy_service` · `unclear`

* new_quote: wants a first quote.
* renewal_or_replacement: an existing policy is expiring, being switched or was non-renewed.
* coverage_question: asks what is or isn't covered / required, not asking to buy.
* document_or_info_provided: sending documents or information that were requested.
* follow_up: chasing an earlier message.
* existing_policy_service: certificates, adding/removing a unit or driver, address changes, claims on a current policy.
* unclear: you cannot tell what the sender wants. For a message with two purposes, choose the main one (your call).

## urgency  (one value per message)
`high` · `medium` · `low`, based on what the sender says (deadline, "ASAP", "no rush"), as you would triage it.

## status_<field>  (12 columns, one value each)
`provided` · `missing` · `unclear` · `not_applicable`

* provided: the message clearly states it.
* missing: the message does not mention it.
* unclear: mentioned but vague or partial (e.g. "a few trucks", only the vehicle types without years).
* not_applicable: does not apply to this sender (in this checklist, only loss runs for a brand-new venture with no insurance history).

Field meanings are in `config/trucking_checklist.yaml` (the `description` of each field is what the checklist counts as provided).
