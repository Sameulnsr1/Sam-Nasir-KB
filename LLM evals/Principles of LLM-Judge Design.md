![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/a919ad231dc05a913b9b5323e3290590eb1aa35e16c557f947e26e5dcc2c91fb/scormcontent/assets/ch6-05-judge-design-princi.jpg)

The decision to build an LLM judge comes from module 5's diagnostic: you've identified a generalization gap where the system works sometimes but not consistently, and the quality criterion requires subjective assessment that code can't capture.

**LLM-as-judge must be used in extremely narrow, well-defined ways. Generic "rate this output 1-10" judges are unreliable.** High-performing teams follow two strict rules:

**Binary Decisions - Judges should provide a single Yes/No decision, not a Likert scale (1-5 rating).**

- Binary forces clarity: either this meets the bar, or it doesn't
    
- Binary labels are significantly easier to validate mathematically (covered later in the calibration section)
    
- Instead of one complex eval with a scale of 1-5, break down evaluations into several binary decisions
    

**Why not Likert - Likert scales (1-5 or 1-10 ratings) have several problems.**

- They’re expensive to align with domain experts because you need agreement on what each number means.
    
- Annotators tend to default to middle values to avoid making hard calls.
    
- They encourage vague, broad criteria like “overall quality” instead of targeted failure modes.
    

**Narrow Scope - A judge should assess one custom trace category at a time.**

If a prompt lists more than a few criteria, it almost always becomes inconsistent. You’ll also lose the ability to diagnose what changed when scores move.

- Focused tasks are easier for an LLM to reason about consistently
    
- Narrower scope improves accuracy (typically 10-15% better than comprehensive judges)
    
- Separate judges enable better root cause analysis (you can see exactly which quality dimension failed)
    
- Easier to calibrate and validate each judge independently
    

## 

### What to Judge: Start With What You Can Teach

Not every subjective category is a good judge candidate. **Start with categories where you can clearly “show” the ambiguous boundary between pass and fail with examples.**

Good candidates tend to look like:

- “Does the response include a concrete next step?”
    
- “Does the response directly answer the user’s question without deflecting?”
    
- “Is the tone appropriate for a professional customer interaction?”
    
- “Does the output make claims that are unsupported by the provided context?”
    

Bad candidates tend to be fuzzy:

- “Is this high quality?”
    
- “Is this delightful?”
    
- “Is this insightful?”
    
- “Is this creative?”
    

If you can’t write down what a pass looks like in plain language and then provide a few crisp examples, you’re not ready to automate it yet.

## 

**Optimizing the Judge Prompt**

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/a919ad231dc05a913b9b5323e3290590eb1aa35e16c557f947e26e5dcc2c91fb/scormcontent/assets/ch6-06-optimizing-judge-pr.jpg)

A judge prompt needs more structure than a typical agent prompt to reduce ambiguity. **A reliable judge prompt usually includes:**

- **A clear role.** Position the model as a reviewer or auditor, not as the original system prompt’s assistant.
    
- **A single evaluation question.** One attribute. One yes/no decision. This should be built from your trace category.
    
- **A concrete standard.** Describe the standard as observable behaviors. Avoid abstract words unless you immediately operationalize them.
    
- **Ambiguous examples** that teach the boundary. Include a few pass examples and a few fail examples, ideally drawn from the same domain and similar tasks. The most valuable examples are the borderline ones where reasonable people might disagree. That’s where the judge is most likely to drift.
    
- **Self-Reflection.** Instructions should include "Think step-by-step" and "Check your reasoning" to force logical analysis before the final score. This significantly improves consistency.
    
- **A strict output format.** Require the judge to return its evaluation as JSON or YAML, ensuring the data is parseable for dashboards and analysis. Include a predictable label and a short justification you can use for debugging.
    

## 

### Running LLM Judges Across a Dataset

The mechanics of running LLM judges mirror running code evals: you execute each judge against each row in your reference dataset and record pass/fail plus the reasoning string. The differences are cost, interpretation, and what "low pass rate" means.

**LLM inference is not free.** A 50-ticket dataset with three judges is 150 model calls per experiment run. At current inference costs, this is manageable — but it means judges should run after code evals pass, not in parallel with every trivial prompt change. Run code evals first, every time. Run LLM judges when the code evals are green, and you're evaluating a substantive change.

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/a919ad231dc05a913b9b5323e3290590eb1aa35e16c557f947e26e5dcc2c91fb/scormcontent/assets/ch6-07-aggregate-llm-judge.jpg)

**Pass rate as a signal, not a verdict.** If the empathy judge pass rate drops from 88% to 62% after a prompt change, something in the new prompt is suppressing the acknowledgment behavior. Before concluding, the agent got worse, read ten failing cases and their reasoning strings.

Example: A pattern like "response opens with account status rather than acknowledgment" across 80% of failure points to a specific prompt clause redirecting agent behavior — not a general quality decline.

**Failure distribution over raw pass rate.** If 90% of actionability judge failures come from tickets classified as Feature Request, the agent is missing next steps specifically for feature escalations, not for billing or technical issues. That's a precise diagnostic. It maps directly to a targeted prompt fix rather than a broad revision.

**Cross-judge correlation reveals root causes.** When the empathy judge and the actionability judge both fail on the same ticket, it usually signals a single failure mode — the agent prioritized efficiency and produced a short, terse response that satisfied neither criterion. Fix the root cause; don't address each judge's failure independently.

**Always compare to your production baseline.** A new version scoring 72% on the factual grounding judge sounds concerning in isolation. If the production baseline was 68%, that's a measurable improvement. Set your baseline on the current production version before you start iterating, and compare every new result against it.

**When a pass rate is unexpectedly low, read the reasoning strings before diagnosing the agent.**

A judge that returns "Fail - response is not a support interaction" on 40% of traces may have a scope problem in its prompt, not a reflection of agent quality. Verify the judge logic before drawing conclusions about the model.

## 

**Failure Modes to Watch For**

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/a919ad231dc05a913b9b5323e3290590eb1aa35e16c557f947e26e5dcc2c91fb/scormcontent/assets/ch6-08-failure-modes.jpg)

Even calibrated judges break in predictable ways. Knowing the failure modes in advance means you can design around them.

**They become easy to game.** If the agent's outputs are also being optimized against the judge, they can drift toward patterns the judge rewards rather than patterns users actually prefer.  
**Example:** after several iterations toward a passing empathy judge, the agent starts opening every response — including routine password resets — with "I completely understand how frustrating this must be." Judge score improves. CSAT doesn't. Users start editing out the boilerplate. The signal: judge metrics improve while user edits or explicit complaints increase.

**They drift when the product changes.** Judges are calibrated on examples from one distribution. When the product expands to new user segments, new domains, or new ticket types, yesterday's examples stop representing today's distribution.  
**Example:** the empathy judge was calibrated on 50 billing tickets from B2C users. The agent expands to handle technical integration questions from developer accounts. Developers don't want empathetic preambles — they want precise technical answers. The judge flags accurate, appropriately concise developer responses as failing, because its examples were all consumer billing interactions. Fix: refresh the example set whenever your user distribution changes materially.

**They break down on genuinely novel outputs.** When there's no stable definition of "good" for a new output type, the judge defaults to rewarding fluency and polish.  
**Example:** the team launches a proactive outreach feature — the agent flags accounts likely to churn and drafts personalized messages. There's no existing example set for this output type. The judge rewards responses that "sound supportive" because that's what its training examples rewarded, even though the success criterion for outreach is specificity and relevance, not warmth.

**They fail when quality requires deep domain expertise.** Judges catch surface-level issues but miss errors that require specialized knowledge.

**They reward style over substance.** Verbose responses that mention timelines, escalation paths, and resolution steps look "actionable" to a model even when they avoid the actual question.  
**Example:** the actionability judge learns to reward responses that include the phrase "will be resolved within 2–3 business days," regardless of whether the response actually addresses the customer's specific issue. Judge score: high. Actual actionability: the customer still doesn't know what to do.

The best judges are paired with disciplined usage: they measure a specific quality boundary you care about, they get their examples refreshed as the product evolves, and they trigger concrete investigation when they fail.

[Lesson 3](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/a919ad231dc05a913b9b5323e3290590eb1aa35e16c557f947e26e5dcc2c91fb/scormcontent/index.html#/lessons/8JhgItJnyjxVWJQScTa2zqLCxiw2v6vs)