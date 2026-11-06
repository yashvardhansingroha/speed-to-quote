# Memo: put an intake copilot in front of our inquiry queue

**To:** Founder  **From:** Operations  **Re:** Speed-to-Quote, a working prototype

> **Every number below comes from SYNTHETIC / DEMONSTRATION DATA** (250 fictional inquiries) and says nothing about our real
> business. This is an independent portfolio project, not a company project.

## The problem

Our edge is speed on complex lines, yet every inquiry starts with a person reading an unstructured message, working out what
is wanted and what is missing, deciding what is urgent, and typing a follow-up. That is slow, and it is where speed is lost
before an underwriter is involved.

## What I built

An engine that turns an inquiry into a record **for a broker to review**: intent, urgency, deadline and a 12-item trucking
checklist, each value backed by a quote from the customer's own words (checked by code). It ranks open work by a transparent
0–100 urgency score and drafts one grouped follow-up that a person edits and approves. It decides nothing about eligibility,
price or coverage, sends nothing, and logs every action.

## What the synthetic analysis shows (illustrative only)

Across 250 synthetic inquiries received 6 Jul to 1 Nov 2026:

- Median first response **13.6 h** (P90 **62.9 h**) over **242** answered inquiries; **1** open inquiry is still awaiting a
  first response and **7** were closed without one.
- **18 of 225** follow-up tasks (**8.0%**) were overdue.
- Submissions arrived **23.4%** complete on average (for human review, not binding).
- **9 of 81** mature quote requests (**11.1%**) bound; **36 of 81** reached a quote.

The history was built with assumptions (faster replies and fuller submissions slightly more likely to progress), so patterns
in it reflect the generator, not a finding. The figures show the analytics layer works, not what our numbers are.

## What I recommend we change (proposals to test, not conclusions)

1. **Work from the ranked inbox**, not arrival order, and read "needs human review" items first.
2. **Send one grouped follow-up** that asks for every missing item at once, from an approved draft.
3. **Sweep overdue follow-ups daily**; give every unanswered inquiry an owner.
4. **Confirm very old open leads once** instead of letting them sit.

## What to test first with real data

1. **Accuracy.** Two brokers hand-label about 100 anonymised messages. The key measure is missing-field **recall**: if the
   tool says something is provided when it is not, we skip a question. Compare the LLM with the rules baseline.
2. **Ranking and baseline.** Does the score order match what a broker would pick first, and what are our real first-response
   time, overdue follow-ups and completeness today?
3. **Time saved per inquiry**, by timing brokers with and without the draft. I make no estimate of this.
4. **Data handling.** Confirm with compliance what may go to an AI provider; until then use only anonymised or consented text.

**Decision needed:** approve a two-week pilot on anonymised inquiries with these measures and a stop rule agreed in advance.
