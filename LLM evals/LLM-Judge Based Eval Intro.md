  

In this module, you will learn how to design narrow, binary LLM-as-Judge evaluators for the quality dimensions that code can’t reach — tone, actionability, and factual grounding.

**An LLM judge is essentially a second model call that evaluates the output of your first model call.**

- It takes a trace as input — or more precisely, the agent's output plus whatever additional context the criterion requires
    
- It returns a binary verdict: Pass or Fail, plus a reasoning string.
    
- The verdict is not deterministic. Run the same judge twice and you may get a different result. That's the tradeoff.
    

You're buying the ability to measure subjective quality, at the cost of the perfect reproducibility that code evals provide.

**The economics are different from code evals.** An LLM judge costs inference on every run. On a 50-ticket dataset with three judges, that's 150 model calls per experiment. This means judges should run after code evals pass — not instead of them — and should be targeted at the specific quality dimensions you've identified through trace analysis.

This lesson covers what LLM judges look like, the two types you'll build, how to apply them to the Support Triage Agent, and how to integrate them into your release process.


### When to use LLM-as-Judge

A large share of quality metrics that users care about is not rule-like:

- A support reply can be factually correct and still feel dismissive in a way that decreases CSAT.
    
- A generated UX can be accurate but still feel like “slop”. A summary can contain the right points and still miss what the user actually needed.
    

In practice, teams hit this ceiling quickly. They can prevent obvious failures, but they still cannot answer questions like:

- Would a user accept this without rewriting it?
    
- Is this specific enough to take action on?
    
- Is it appropriately asking for permissions - or doing too much or too little?
    

This is where LLM-as-Judge evaluators enter.

**The model now also becomes your evaluator: reasoning about quality instead of just generating content.**

But building a reliable custom judge is unglamorous, foundational work that teams are often tempted to skip in favor of "out-of-the-box" evals - that are rarely actionable.

## 

**Common Categories for LLM-Judge Evals**

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/a919ad231dc05a913b9b5323e3290590eb1aa35e16c557f947e26e5dcc2c91fb/scormcontent/assets/ch6-01-llm-judge-categorie.png)

Not every subjective quality property is equally well-suited to LLM judging. Four categories show up across products and consistently require a judge rather than code.

**Tone and empathy gaps**  
These appear when the agent's response is technically correct but fails the human interaction. A billing response that resolves the issue but ignores the customer's frustration. A support reply that's accurate but clinical when the situation calls for warmth. These properties have no regex equivalent.

**Directness and actionability failures**  
These surface when the agent hedges, over-qualifies, or provides information without telling the user what to do next. "Your account may have been affected," where "Your account was charged twice — here's how we'll fix it" is what's needed. Code can check for the presence of a next step, but only a judge can assess whether it's genuinely actionable.

**Factual grounding & memory problems**

In search-based systems, a lack of grounding results in the agent making claims that contradict retrieved context, overstating confidence, or omitting information that was in the retrieved documents and should have been included. Evaluating these requires reading both the output and the source material — called a reference-based LLM judge.

**Semantic completeness & context management**  
This matters when outputs must address all parts of a multi-part question or request. Did the summary cover all the key decisions from the document? Did the recommendation address both the immediate issue and the underlying cause? Did any context get lost after compaction or truncated? Pattern matching can check for keywords. A judge can determine whether the substance is there.

## 

**What an LLM Judge Looks Like**

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/a919ad231dc05a913b9b5323e3290590eb1aa35e16c557f947e26e5dcc2c91fb/scormcontent/assets/ch6-02-llm-judge-anatomy.jpg)

**Every LLM judge has the same four-part anatomy.**

**Inputs: the trace plus an optional reference.** An LLM judge takes two kinds of inputs: the agent's output (the trace) and, optionally, a reference — which is either the original user input, a ground truth answer, or retrieved context. What you pass in determines what the judge can measure:

- A judge who only sees the output can evaluate tone, format, and directness.
    
- A judge who sees the output and the user's original query can evaluate relevance and responsiveness.
    
- A judge who sees the output alongside a reference document can evaluate factual accuracy and grounding.
    

**The judge prompt: role, criterion, standard, examples.** The prompt positions the model as an auditor, not an assistant. It defines the single criterion being evaluated, provides observable pass/fail standards, and includes boundary examples that teach the judge where the line is. A vague judge prompt produces unreliable scores.

**The LLM call: reasoning before verdict.** Use a reasoning model where possible. The chain-of-thought that precedes the verdict is your debugging interface. When the judge disagrees with your human reviewer on trace #23, reading the judge's reasoning tells you whether the judge misunderstood the criterion, hit an edge case the examples didn't cover, or is actually right and the human reviewer was inconsistent.

**The output: Pass or Fail, plus a reasoning string.** The reasoning string is not optional. "Fail" with no explanation is useless at scale. "Fail — response opens with account status rather than acknowledging customer frustration; no empathetic framing before resolution" tells you exactly what pattern failed and where to look in the prompt.

### 

### **Example 1: Tone check for Support Triage**

The trace analysis for the Support Triage Agent surfaced a specific failure pattern: on high-severity billing tickets, some responses were

- Technically correct but tonally flat:
    
- No acknowledgment of frustration.
    
- No urgency.
    

Code can confirm that a response includes the words "I understand" — but it can't tell whether that acknowledgment is specific to the customer's situation or a generic filler.

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/a919ad231dc05a913b9b5323e3290590eb1aa35e16c557f947e26e5dcc2c91fb/scormcontent/assets/ch6-03-empathy-judge-promp.jpg)

The practical threshold: if a reviewer needs to hold the original user request in mind and compare it to the output — or reason about tone, intent, or quality — that's an LLM judge. If you can write the success condition as a Python function in under ten lines, it belongs in code.

### 

### Reference-Free vs Reference-Based Judges

Every LLM judge falls into one of two categories, and the distinction shapes what evidence you need to build it.

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/a919ad231dc05a913b9b5323e3290590eb1aa35e16c557f947e26e5dcc2c91fb/scormcontent/assets/ch6-04-reference-free-vs-b.png)

**Reference-free judges** evaluate the output on its own merits. They don't need to know what the correct answer looks like — they assess properties of the output itself:

- Is it direct?
    
- Is the tone appropriate?
    
- Does it include a concrete next step?
    
- Does it make an unsupported claim?
    

Reference-free judges can be built as soon as you have representative traces. You don't need a labeled dataset.

**Reference-based judges** compare the agent's output against something external — the original user query, a ground truth answer, or retrieved context chunks. The hallucination guard from the previous section was a code-based reference-based eval: it compared output ticket IDs against input ticket IDs. Reference-based LLM judges do the same thing for properties that can't be checked with a regex:

- Does the response accurately reflect the content that was retrieved?
    
- Does the category label match what a domain expert would assign?
    

**The practical implication:** reference-free judges can be built immediately from any representative traces. Reference-based judges require ground truth — either expert-labeled outputs or access to the retrieved context at inference time.

**If you're early in your eval development and don't yet have labeled data, start with reference-free judges.**

Use them to build your dataset. Add reference-based judges once you have a reliable ground truth to compare against.