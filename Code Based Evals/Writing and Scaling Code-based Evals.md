Four trace property types are ideal for code-based evaluation: structure/format, presence and coverage, tool call success, and threshold checks.

## 

Structure and format

AI outputs are often required to conform to a schema: JSON with specific fields, a response that includes required sections, and a character count under a UI limit. Code evals are the right tool here. Schema validation libraries exist for this. **These checks are especially important when your AI system feeds downstream pipelines — a malformed output that passes a vibe check can silently break production.**

**For the Support Triage Agent:** Does the JSON response include both a category label and a priority level? Are all required fields present? Is the summary under 500 characters?

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776312000/ZvsokyZ4rwMWELBdgjb8bQ/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-04-check-response-sche.jpg)

## Presence and coverage

Does the output contain specific keywords, phrases, or identifiers? Does it reference the correct product name, the right ticket ID, and the required policy language? **These are string-matching and regex problems. They're cheap to run and straightforward to debug.**

A concrete example: evaluating whether a conversational agent asks open-ended follow-up questions. The check is a keyword scan against a target set. If any open-ended question marker appears, the eval passes.

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776312000/ZvsokyZ4rwMWELBdgjb8bQ/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-05-check-open-question.jpg)

**Similarly, for the Support Triage Agent:** does the response reference only ticket IDs that appeared in the original input? Does it avoid inventing IDs that weren't there?

## Tool call sequencing

**If your agent uses tools — retrieval, database lookups, API calls — you can check whether it called the right tools, in the right order, with the right parameters.**

This is one of the most underused categories of code-based eval and one of the most valuable. Tool call logs are structured data. They're easy to check programmatically.

**For the Support Triage Agent:** Did the agent call Subscription_Check before Resolution_Step? Did it pass the correct user_id? Did it avoid triggering Escalation for tickets below the severity threshold?

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776312000/ZvsokyZ4rwMWELBdgjb8bQ/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-06-check-tool-call-ord.jpg)

## Threshold checks

**Latency, cost, token count, confidence scores — any numeric property can be checked against a threshold.** These are simple comparisons, but they catch important regressions. A new prompt version that passes every quality check but doubles response latency is still a problem.

**For the Support Triage Agent:** Is end-to-end latency under 2 seconds? Is the response within the character limit? Is the cost-per-ticket within budget?

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776312000/ZvsokyZ4rwMWELBdgjb8bQ/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-07-check-sla-latency.png)

### Case Study: 3 Code Evals for the Support Triage Agent

The trace analysis from Module 1 revealed a set of error patterns. Three of them have clear, deterministic success conditions. They become the first three code evals in the eval suite.

### **Eval 1: Category label format check**

The prompt specifies that every response must include exactly one of: Technical, Billing, or Feature Request. The generalization gap: Some responses describe the category rather than using the canonical label. "This is a billing-related question" passes a vibe check but breaks the downstream routing system that parses the exact label.

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776312000/ZvsokyZ4rwMWELBdgjb8bQ/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-08-case-study-category.png)

**Passing case:** "Category: Billing — Customer is disputing their March invoice." → True, "Single valid category: Billing."

**Failing case:** "This appears to be a billing-related issue." → False, "No valid category label found in output."

**Diagnostic value: When this eval fails at scale, it tells you which prompt version introduced the paraphrasing behavior.** It also shows whether specific input types — short tickets, ambiguous tickets, non-English inputs — are disproportionately likely to produce unlabeled outputs.


### **Eval 2: Hallucination guard — no invented ticket IDs**

The trace analysis found a specific failure mode: when summarizing a ticket, the agent occasionally references a ticket ID that wasn't in the input. This is a hallucination. It can't be detected by reading the output alone — you have to check the output against the input.

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776312000/ZvsokyZ4rwMWELBdgjb8bQ/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-09-check-invented-ids.jpg)

**Passing case:** Input mentions TKT-00421. Output references TKT-00421. → True

**Failing case:** Input mentions TKT-00421. Output references TKT-00422. → False, "Invented ticket IDs in output: {'TKT-00422'}."

**Diagnostic value: Hallucination of ticket IDs tends to cluster around specific agent behaviors** — usually when the model is asked to generate a summary that references prior context it doesn't actually have. The eval identifies exactly which inputs trigger this.


### **Eval 3: Response latency SLA**

The product requirement is a sub-2-second classification. After a prompt update that added additional reasoning context, latency on the 50-ticket dataset increased from an average of 1.1 seconds to 2.4 seconds. The code eval caught this before release.

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776312000/ZvsokyZ4rwMWELBdgjb8bQ/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-10-case-study-latency.jpg)

These three evals cover format correctness, hallucination risk, and performance. They run on every prompt change. **When all three pass on the full dataset, the change is a candidate for release. When one fails, the failure message tells you exactly what to investigate.**

**Pitfall to avoid:** Don't confuse a green eval suite with a good agent. These three evals check specific properties you chose to measure.

**They don't check whether the agent is helpful, coherent, or appropriately empathetic. Code evals are the floor, not the ceiling.**

The full eval strategy — LLM judge, human review, user feedback — addresses what code can't reach.


**Running Code Evals Across a Dataset**

PMs should lead on prioritizing and framing evals. You then work with your engineering team to **run them systematically across your reference dataset and read the results.** The mechanics are simple.

1. You have a dataset (eg, the 50-ticket golden set from Module 1).
    
2. You run each eval function against each row.
    
3. You record pass/fail and the reason string for every combination.
    
4. Then you look at the aggregate.
    

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776312000/ZvsokyZ4rwMWELBdgjb8bQ/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-11-aggregate-metrics.jpg)

**Pass rate: the percentage of rows that pass the eval.**  
Your baseline pass rate on the current production version is the reference point. A new prompt version that drops pass rate from 92% to 78% on the category label eval failed — even if every individual output looks fine in spot checks.

**Failure distribution: Are failures concentrated on specific input types?**

If 80% of hallucination eval failures come from tickets under 20 characters, the eval has given you a precise diagnostic. Short, underspecified inputs are the failure mode.

**Cross-eval correlation:** when the category label eval and the hallucination eval both fail on the same row, that's a signal about input complexity, not two independent problems. Look at the rows where multiple evals fail simultaneously.

**When the pass rate is unexpectedly low, check the eval logic first:**

- A pass rate of 0% almost always means a bug in the eval function, not a fundamentally broken agent.
    
- Read five failure reasons. If they're all "No valid category label found" on outputs that clearly contain category labels, your string matching has a case sensitivity issue.
    
- Fix the eval before you investigate the agent.
    
- Once the eval logic is verified, the low pass rate is a diagnostic signal. It tells you where to focus.
    


### Code Evals in Your Release Workflow

The goal is to run code evals automatically on every prompt change or model upgrade before anything ships.

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776312000/ZvsokyZ4rwMWELBdgjb8bQ/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-12-release-workflow.jpg)

**The setup:** you have a reference dataset, a pass rate baseline for each eval on your current production version, and an explicit threshold for each eval.

**When you make a change, you run the eval suite, compare the results to the baseline, and check against the thresholds.**

- Set thresholds before you start iterating, not after. The decision of whether 82% is good enough should not be made after you've already seen the number and want to ship.
    
- Decide in advance: the category label eval must be above 90% for a prompt change to move forward. Write it down. Treat it as a hard gate.
    
- The threshold varies by what you're shipping. A major architectural change — new retrieval method, new model family — warrants a higher bar and a larger reference dataset. A small prompt tweak can use a narrower margin.
    

The principle is consistent: the eval suite is a release gate, not a post-mortem tool.

**Quick rule of thumb**— If you can write the success condition as a Python function in under ten lines, it belongs in code. If expressing the success condition requires understanding intent, evaluating tone, or reasoning about quality, that's an LLM judge. We’ll dive into this in our next module.

[Lesson 3](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776312000/ZvsokyZ4rwMWELBdgjb8bQ/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/index.html#/lessons/gu85S81Dnn3T3fZbNAddJ68iy5uNS2j4)