In this module, you will learn how to write deterministic, code-based evaluations that run automatically on every prompt change and serve as the non-negotiable foundation of your eval suite.

As we learnt in the previous lesson, **the default path for automated evaluation should always be code-based because it's faster, cheaper, and 100% reproducible (deterministic).**

## 

**Common Categories for Code-based Evals**

When we show up to the present moment with all of our senses, we invite the world to fill us with joy. The pains of the past are behind us. The future has yet to unfold. But the now is full of beauty simply waiting for our attention.

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-01-code-eval-categorie.jpg)

Certain kinds of trace codes show up across products and are strong candidates for deterministic checks.

- **Structure and format** **issues** appear whenever outputs are consumed by other systems. Missing fields, malformed JSON, or invalid schemas are easy to detect and expensive to ignore.
    
- **Presence and coverage** **gaps** show up when outputs omit required elements. A pitch that never mentions market size. A summary that doesn’t cite sources. A recommendation that lacks a next step.
    
- **Tool call failures** become visible in multi-step systems. Agents call tools that don’t exist, omit required parameters, or invoke steps out of order. These failures are mechanical, not conceptual.
    
- **Search quality problems** surface in RAG and GREP-based systems. If the right documents weren’t retrieved, the generation step never had a chance. Measuring retrieval accuracy directly is often easier than judging the final answer.
    

## 

**What exactly is a code-based eval?**

**A code-based evaluator is a Python function**. It takes a trace as input - or more precisely, the output string, tool call log, or structured response from your AI system - and returns a pass or fail. No LLM call. No probability distribution. One deterministic result, every time. That determinism is the point.

**Code-based evals are the bedrock of a reliable eval suite. They run on every prompt change, every model upgrade, every release candidate.** They tell you immediately when something that was working has broken. When you have them, you can move fast. When you don't, you're relying on vibe checks to catch regressions.

**The economics matter too. Code evals are fast and cheap — deterministic checks cost nothing per run compared to LLM inference.** They can run in the critical path of your production pipeline to catch failures in real time, not just in offline testing. And they're 100% reproducible: the same input produces the same result every time, which means your eval results are comparable across versions.

This chapter covers what code-based evals look like, which trace categories they handle best, how to apply them to the Support Triage Agent, and how to build the infrastructure to run them across a dataset.

## 

**What Code-Based Evals Look Like**

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-02-eval-anatomy.jpg)

Every code-based eval has the same three-part anatomy.

**Inputs:** **the full trace.** Usually, the complete response from your AI system — an output string, a JSON response, the sequence of tool calls, or some combination. Some evals also take the original user input to check context-dependent properties (did the output reference the correct ticket ID?).

**The check: the logic.** A condition, a pattern match, a schema validation, a count, a threshold comparison. Whatever you're measuring, it can be expressed precisely in code without requiring language understanding.

**The output: pass or fail, plus a reason string.** The reason the string is not optional. When an eval fails on trace #38 of 50, you need to know why without reading the full trace. "Output contains no valid category label" is useful. "False" is not.

### 

**Example 1: Category checking for Support Triage** 

The Support Triage Agent is supposed to classify each ticket into exactly one of three categories:

- Technical
    
- Billing, or
    
- Feature Request.
    

This code-based eval checks that every response includes the correct label — not a paraphrase, not "billing issue," not a free-form description, but the exact string.

![](https://cdn5.dcbstatic.com/files/r/e/reforge_docebosaas_com/1776373200/3sM3VcN3GxgzSUT9M3qpBw/scorm/37b9f317e7449d02f257a43585e793573602ce0f4a276447387abe98047fb69c/scormcontent/assets/ch5-03-check-category-pres.jpg)

This is a code-based eval. It is eight lines long. It runs in microseconds. It gives an unambiguous result on every ticket in your dataset.

**The practical threshold: if you can write the success condition as a Python function in under ten lines, it belongs in code.**

Every hour you spend writing an LLM judge for something code could check is an hour you didn't spend on the genuinely hard problems.