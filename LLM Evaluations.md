# Links
- https://blog.promptlayer.com/llm-eval-framework/?utm_source=chatgpt.com 
- https://www.datadoghq.com/blog/llm-evaluation-framework-best-practices/?utm_source=chatgpt.com&__readwiseLocation= 
# Metrics
* F1-score, Precision, Recall (for binary outputs) 
* BLEU/ROUGE scores
* Perplexity
* LLM-as-judge
* Human review

# Offline Evaluation Set
Assemble "golden dataset". Should include:
- Typical inputs
- Edge cases
- Adversarial prompts

## Annotation
- Expert Labelers
- LLM-assisted workflows
- Target high inter-annotator agreement

## Faithfulness Evaluations

[Faithfulness](https://aclanthology.org/2024.findings-acl.19/) evaluations use a secondary LLM to test whether an LLM application’s response can be logically inferred from the context the application used to create it—typically with a RAG pipeline. A response is considered faithful if all its claims can be supported by the retrieved context, while a low faithfulness score can indicate the prevalence of hallucinations in your RAG-based LLM app’s responses. Open source providers like [Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) and [DeepEval](https://docs.confident-ai.com/docs/metrics-faithfulness) offer out-of-the-box faithfulness evaluators you can integrate into your experimentation and monitoring. Typically, a faithfulness evaluator will execute the following steps:

1. Break down the response into discrete claims.
2. Ask an LLM whether each claim can be inferred from the provided context.
3. Determine the fraction of claims that were correctly inferred.
4. Produce a score between 0 and 1 from this fraction.

# Online Metrics

## User Experience Evaluation
- Negative Sentiment
	- Flag user frustration or application misuse
	- Break down each user response into discrete statements
		- For each statement, ask secondary LLM if the statement is negative, neutral, or positive
- Topic Relevancy
	- Ask secondary LLM if answer falls within defined domain boundary
	- Can also check relevancy of *question*: if question falls outside domain, can put guardrail in place. 

# Safety and Security
- Match against keywords
- LLM-as-a-judge: 
	- open-source toxicity detection [from Meta](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) 
	- open-source jailbreak eval toolkit available