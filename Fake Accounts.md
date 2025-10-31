# Generic Framework
## Frame as ML Problem
We treat this as an unsupervised clustering problem, where we want to identify clusters of similar accounts based on behavioral, device, profile, or content signals. These clusters might be **bots**, **spam**, or other **fake accounts**. Assume that we **don't have access to moderators**.m 

## Data Preparation
- $\text{Reciprocity}(u) = \frac{\text{\# of mutual connections}}{\text{\# of people user follows}}$ : fake/bot accounts tend to have low reciprocity as they follow many accounts but not many follow them back. 
	- Note that on Tiktok, this is less applicable.
- $\text{Ratio}(u) = \frac{\text{\# of followers}}{\text{\# of following}}$ : Fake/bot accounts will have ratio << 1 and below that of normal accounts

| Input Data         | Details                                                                        |
| ------------------ | ------------------------------------------------------------------------------ |
| Profile Metadata   | Registration time, Bio text, User avatar                                       |
| Behavioral Signals | Number of videos viewed, Number of uploads in past 1/3/7d                      |
| Content Signals    | Image embeddings, Text embeddings                                              |
| Social Graph       | Number of followers, Number of followed, Reciprocity, Follower/Following Ratio |
- Standardize numeric features, encode categorical features
- Run dimensionality reduction with PCA or UMAP
## Modeling and Training
- Modeling
	- DBSCAN/HDBSCAN on continuous signals
	- Louvain/Leiden community detection on categorical signals
- Offline evaluation
	- Overlap with user reports as "recall"
	- Cluster validation metrics, e.g., silhouette score
	- Take random sample of known good users - what % are clustered? 
	- Take random samples from largest K clusters
	- For Leiden, what are the main signals of connection?

## Deployment and Serving
- Deployment
	- Run offline daily or hourly depending on downstream/business needs
- Online evaluation
	- A/B tests
	- Target metrics:
		- User reports (# reports per 1k active users)
		- Repeat offender rate (% users who are reported multiple times)
		- Suspicious follow ratio
		- Engagement farming - % of likes/comments coming from new accounts
	- Control metrics:
		- User appeals (though need to be careful about fake appeals)
		- Decrease in recommendation quality




# "Unveiling Fake Accounts at the Time of Registration", TenCent
[UFA paper](papers/ufa_paper.pdf)  

## Methods Section Summary

The UFA (Unveiling Fake Accounts) system consists of four main components designed to detect fake accounts at registration time in an unsupervised manner:

### 1. Feature Extraction

Based on a measurement study of WeChat registration data, the authors identified that fake accounts exhibit outlier registration patterns. They extract features from registration attributes including:

- **IP address, phone number, device ID, MAC address** (Pre_B features - higher frequency indicates more anomaly)
- **OS version, WeChat version, time anomaly, location anomaly** (Pre_A features - lower frequency indicates more anomaly)

Key observation: Fake accounts cluster on these patterns (same IPs, outdated versions, midnight registrations, location inconsistencies).

### 2. Unsupervised Weight Learning

- **Registration-Feature Bigraph Construction**: Creates a bipartite graph where nodes represent either registration accounts or features, with edges connecting accounts to their features.
- **Weight Initialization**: Uses a statistical method to assign initial weights to features based on feature ratios and mode features, without requiring labeled data. The weight formula considers both the deviation from the mode feature and the baseline anomaly of the feature prefix.
- **Weight Update**: Applies linearized belief propagation to iteratively update node weights by propagating information across the graph structure, capturing relationships between features and accounts.

### 3. Registration Graph Construction

Maps the registration-feature bigraph into a registration graph where:

- Nodes represent registration accounts only
- Edges connect accounts with similarity above a threshold
- Similarity is calculated as the sum of weights of shared features
- Result: Fake accounts become densely connected; benign accounts remain sparse

### 4. Fake Account Detection

Uses the Louvain community detection algorithm to:

- Cluster densely connected nodes into communities
- Flag all accounts in communities exceeding a size threshold (default: 15) as fake accounts

The key innovation is detecting fake accounts without manual labels by exploiting the clustering behavior of fake accounts on outlier registration patterns.



# Account Takeover

How to build Account Takeover model? What are some of the considerations to reduce ATO at account login? 

## Clarifications
- What do we mean by ATO? 
- What are our labels? 

## ML Task
- Supervised Learning based on end-user feedback
- Goal is to beat user feedback

## Data Preparation
Note: perception is $T+N$, defense is point-of-login
- Features
	- Sequence data => Perception model: 
		- Sequence model is trained on 
	- Login data => Defense model
		- Device signals: device version
		- Network signals: IP address, user_agent 
		- Behavioral signals: behavioral features
		- Local anomaly scores: can also flag anomalies based on, e.g., historical IP addresses. Calculate z-score. => Defense
- Labels
	- Positive label: user feedback: account has been hacked
	- Negative label: other accounts => cleansing process => not fake accounts, not banned. 

* Important tradeoff: If you increase $N$, you get more feature data but the higher the chance the ATO happens and you get negative user feedback

## Serving/Deployment
* Online evaluation
	* User feedback rate (ATO rate)
	* User appeals
* Want to maximize recall at set precision
## Punishments
- Hard block
- MFA
- Captcha

