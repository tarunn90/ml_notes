- https://www.youtube.com/watch?v=XN2ymraj27g 

# Framework
## 1. Understand the Problem (5 min)
- Take 5 minutes to really understand the question
- Limit the amount of clarifying questions: make assumptions
	- E.g., "we'll assume Facebook has ~3 billion users"
- Is there a baseline model we can compare against? Heuristics? 

## 2.  Build a high-level design (5 min)
- Translate the problem into ML: is it classification, regression, recommendation, or ranking?
	- If it is classification, is it binary, multiclass, or multitask? 
- This is your system blueprint: no details, just high-level
- Keep it very high-level: instead of "XGB", say "model"

## 3. Data Preparation (8-9 min)
- Understand your labels and where they're coming from
- Understand your features 
- Figure out how to clean and standardize the data: normalization, etc.
- Figure out how to represent the data: one-hot-encoding, embeddings, etc. 

## 4. Modeling and Training (15 min)
- How to split the data? 
	- By time?
	- Stratify by class?
- Modeling
	- Which algorithm to use? 
- Offline evaluation metrics
	- Which metric? Pros and cons? 
	- Compare against simple baseline (e.g., predict most prevalent class, recommend most popular videos)
- Dealing with overfitting
- Imbalanced classes
	- Over/under-sampling
	- Weight the classes in loss function
	- SMOTE

## 5. Serving (5 min)
- Online inference or batch inference?
	- Are there relevant real-time features?
	- What is the latency requirement? 
- Online evaluation
	- Recsys - CTR, watch time
	- Harmful content - prevalence, appeals, user reports
- Rollout strategy
	- Shadow deployment
	- A/B test
	- Canary
- Autoscaling
	- Kubernetes has load balancing and health checks
- For RecSys - explore vs exploit
- Retraining
	- Cadence?
	- Continual learning? 
	- Active learning? 
- Deployment
	- On-cloud vs device
	- Model compression: model pruning, quantization
- Model Registry
	- Track model versions and required metadata to recreate model
- Monitoring
	- Online accuracy
	- Label shift or Feature shift causing train-serving skew: set alerts. 
		- Might need to retrain more frequently
	- IFF we detect a problem from coarse-grained metrics, we switch to fine-grained metrics

## 6. Ask the Interviewer Questions! 


# Common Domains

- 1. Recommendation Systems
- 2. Harmful Content

For smaller companies, it will be more targeted to their product. E.g., vision or NLP. 