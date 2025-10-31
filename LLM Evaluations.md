# Links
- https://blog.promptlayer.com/llm-eval-framework/?utm_source=chatgpt.com 
- https://www.datadoghq.com/blog/llm-evaluation-framework-best-practices/?utm_source=chatgpt.com&__readwiseLocation= 
- https://chatgpt.com/share/6902ebf9-eb80-8003-989a-dbb4f92dd9a6


# Metrics
* F1-score, Precision, Recall (for binary outputs) 
* BLEU/ROUGE scores
* LLM-as-judge
* Human review
* Perplexity (only relevant during model pre-training/fine-tuning)



# Offline Evaluation Set
## 1. Golden Dataset
Assemble "golden dataset" of structured outputs. Should include:
- Typical inputs
- Edge cases
- Adversarial prompts
- Correct answers for structured queries
- F1 score on extracted entities


## 2. Annotation
- Expert labelers grade the LLM outputs against a rubric including:
	- Medical accuracy
	- Empathy
	- Clarity
	- Compliance
	- Safety/Security
- Target high inter-annotator agreement


## 3. Adversarial Testing
"Red-team" testing


## 4. G-Evaluation
**G-Eval:** Generative Evaluation. Provides a middle ground between human evaluation (expensive but slow) and automated metrics like BLEU or ROUGE (fast but shallow, surface-level). **We use another LLM (the "evaluator LLM") to evaluate our chatbot**. 

1. Define evaluation criteria, e.g., factual accuracy, relevance, coherence, safety, empathy
2. Design a structured prompt for the evaluator LLM, e.g.: 
```markdown 
You are an expert judge evaluating chatbot responses for healthcare support.
Rate each response from 1–5 on the following:
- Accuracy: Is the information correct?
- Clarity: Is it easy to understand?
- Empathy: Is it sensitive and appropriate?
Provide a brief justification for each rating.
```

3. Provide model output (and, perhaps answer from reference or baseline) 
4. Evaluator LLM scores each output using the rubric
5. Aggregate scores

**Risk Areas:** 
- Using same model family to score itself
- Prompt sensitivity

**OpenAI Evals** is a framework for running G-Evals.

### 4a. Faithfulness Evaluations
While G-Evaluation is an *evaluation method*, [Faithfulness](https://aclanthology.org/2024.findings-acl.19/) is *an evaluation metric*. We use a secondary LLM to test whether an LLM application’s response can be logically inferred from the context the application used to create it—typically with a RAG pipeline. A response is considered faithful if all its claims can be supported by the retrieved context, while a low faithfulness score can indicate the prevalence of hallucinations in your RAG-based LLM app’s responses. Typically, a faithfulness evaluator will execute the following steps:

1. Break down the response into discrete claims.
2. Ask an LLM whether each claim can be inferred from the provided context. E.g., "How true is the model output to the facts in the source?" 
3. Determine the fraction of claims that were correctly inferred.
4. Produce a score between 0 and 1 from this fraction.

Note that Faithfulness can also be measured by humans or rule-based methods (NER, fact overlap). 




# Online Metrics

## 1. Conversation-Level Metrics
- CSAT User Satisfaction Score: $\frac{\#(\text{satisfied responses})}{\#(\text{total responses})}$ 
- Resolution rate: % of sessions resolved without human-in-the-loop
- Average conversation length
- AB test against baseline


## 2. User Experience Evaluation
- Negative Sentiment
	- Flag user frustration or application misuse
	- Break down each user response into discrete statements
		- For each statement, ask secondary LLM if the statement is negative, neutral, or positive
- Topic Relevancy
	- Ask secondary LLM if answer falls within defined domain boundary
	- Can also check relevancy of *question*: if question falls outside domain, can put guardrail in place. 


## 3. Safety and Security
- Match against critical keywords
- Handoff rate for critical topics
- LLM-classifier monitors safety
- PHI leakage, compliance


## 4. QA Sampling with Human-in-the-Loop
- Score on offline rubric from above