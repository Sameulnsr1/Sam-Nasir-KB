→ [https://lnkd.in/g_W6rMPP](https://lnkd.in/g_W6rMPP) — CLI for testing prompts, agents, RAGs. Used by OpenAI and Anthropic
→ [https://lnkd.in/ghm2KjCV](https://lnkd.in/ghm2KjCV) — pytest-like unit testing for LLM apps
→ [https://lnkd.in/gAPdCRzh](https://lnkd.in/gAPdCRzh) — RAG-specific evaluation: faithfulness, relevancy, context precision
→ [https://lnkd.in/gS9QB3bM](https://lnkd.in/gS9QB3bM) — open-source LLM tracing, evals, prompt management
→ [https://lnkd.in/g9wPBReC](https://lnkd.in/g9wPBReC) — AI observability with OpenTelemetry-native tracing
→ [https://lnkd.in/gWkztfUX](https://lnkd.in/gWkztfUX) — open-source LLM evaluation and tracing platform

- Evals Overview
	- https://hamel.dev/blog/posts/evals-faq/
		- Golden Dataset Review
			-  Validate that proposed codes actually map to patterns you observed — not just what sounds plausible
			- Decide severity ranking across dimensions (which failures matter most to your product)
			- Catch codes that are too generic — "quality" or "accuracy" as a category is the LLM defaulting to vagueness
			- Make final naming decisions — these names become shared vocabulary across your team and tooling
			- Test rubrics against borderline examples yourself — this is where you'll find the gaps the LLM missed
- LLM - As a Judge
	- https://hamel.dev/blog/posts/llm-judge/
		- Should I break out each Axial Code by Feature as well? (I.e. What workflow was it solving for i.e. Teams, Jira, System, Airtable, etc..)
		- We should add critques as part of out AI Evals Reference. It's not included anymore
			- For **passes**, we explain why the AI succeeded in meeting the user’s primary need, even if there were critical aspects that could be improved. We highlight these areas for enhancement while justifying the overall passing judgment.
			- For **fails**, we identify the critical elements that led to the failure, explaining why the AI did not meet the user’s main objective or compromised important factors like user experience or security.
		- Most importantly, **the critique should be detailed enough so that you can use it in a few-shot prompt for a LLM judge**. In other words, it should be detailed enough that a new employee could understand it. Being too terse is a common mistake.
		- Should we define what each Axial Code Means?
		