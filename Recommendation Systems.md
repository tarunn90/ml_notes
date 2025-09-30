
# Links
- https://developers.google.com/machine-learning/recommendation/dnn/training 
- https://medium.com/data-science/recommender-systems-a-complete-guide-to-machine-learning-models-96d3f94ea748 


# Common design pattern:

(1) Retrieval -> (2) Scoring -> (3) Re-ranking

(1) Retrieval: generate smaller subset from population of items. Fast and scalable but not precise (high recall, low precision)

(2) Scoring: use precise model to score candidates

(3) Re-ranking: Rank by score but also apply filters, e.g., user explicitly dislike or boost newer items. 

# Embeddings

# Similarity Measures
- Cosine similarity
- Dot product
	- Equal to cosine similarity if the vectors are normalized (unit-length)
- Euclidean distance
	- 1:1 with cosine similarity if the vectors are normalized

More popular items may have larger norms (because they get more gradient updates and move farther from the origin during training) and this will influence the dot product similarity, for better or worse.

# Content-Based Filtering

Uses *similarity between items* to recommend items similar to what the user has liked. 
* Each row in the item matrix represents an item's features. Might be binary encoded. 
* We compare each item row to the user's row with the same features. 


![[Pasted image 20250923110510.png]]

- Pros: 
	- Doesn't need data from other users
	- Can capture user-specific interests
- Con:
	- Cold-start problem if new user 
	- Feature representation requires hand-engineering
	- Can't recommend *new* interests for user since it only works off prior interests

# Collaborative Filtering

Uses *similarities between users* to recommend items. If user A is similar to user B and user B liked item 1, we recommend item 1 to user A. 
- Features can be hand-engineered or learned embeddings
- **Feedback Matrix**: each row is a user, each column is an item, also called a **user-item matrix**
- Features can be *explicit* (users' ratings of items) or *implicit* (if user watches or engages with movie) 

Note that this is user-user collaborative filtering, but there is also item-item collaborative filtering:

![[Pasted image 20250929183846.png]]

## Matrix Factorization

Given a feedback matrix which is $A \in \mathbb{R}^{mxn}$ , $m$ = number of users, $n$ = number of items, a matrix factorization learns:

- $U \in \mathbb{R}^{mxd}$ : user embedding matrix
- $V \in \mathbb{R}^{nxd}$ 

Then the product $U V^T$ is an approximation of $A$. 
- $U$ and $V$ are more compact representations than $A$: $A$ has $O(mn)$ entries while $U, V$ have $O(d(m+n))$ entries. This is very helpful when $d << n$ or $d << m$. 
- $U$ and $V$ therefore learn latent structure in the data, similar to PCA

How do we learn $U$ and $V$? We can use squared error between $A$ and $UV^T$: 

$$\min_{U,V} \sum_{(i,j)} (A_{ij} - \langle U_i, V_j \rangle)^2$$

Then we just learn $U, V$ to minimize the squared loss using the observed data.

Note that the prediction for a specific pair item $i$ and user $j$ basically boils down to the dot product of $U_i, V_j$. 

Note that the **outer product** of $U_i, V_j$ is a matrix where entry $i,j$ tells you how well user attribute $i$ aligns with item attribute $j$. 

However, there's a major problem here. If we only use the observed $(i,j)$ pairs , then the model can spit out any random or bad values for *unobserved* pairs $(i,j)$ without affecting the loss function. This will make the model overfit and generalize poorly. There are several approaches to address this:
1. Add regularization to keep predictions reasonable and help generalization.
2. Label the unobserved entries as some tunable value $w_0$ which is probably close to 0 (assume we have a scale $[-1,1]$ so $0$ is "indifferent"). We can use held-out data to learn the value of $w_0$. 

Pros: 
- Rather than needing domain knowledge, we learn embeddings
- Rather than only relying on user's prior interests, we can show them items, increasing the diversity of recommendations
Cons:
- Cold-start problem for items
	- Can be approximated with heuristics (e.g., average of embeddings from same creator)
- Cold-start problem for user
	- Can ask user for interests on registration
- Hard to include "side features", e.g., age, country

# Deep Neural Networks

Say we treat the problem as a multiclass classification prediction problem where the *input is the user query* and the *output is a probability vector with dimension = number of items*. Then the model output becomes the probability vector that a given user interacts with all of our items. 

- Input features can include dense features (watch time, time since last watch) and sparse features (watch history, country). 
	- We can also include "side features", e.g., country, age. 
- Let $\psi(x) \in \mathbb{R}^d$ be the output of the last hidden layer. 
- Then the model maps: $\psi(x) \rightarrow \hat{p} = h(\psi(x)V^T)$ where:
	- $\hat{p}$ is the probability vector
	- $h$ is the softmax function
	- $V \in \mathbb{R}^{nxd}$ is the matrix of weights of the softmax layer, which maps from scores to probabilities. $V$ is kind of our item embedding matrix now.  ![[Pasted image 20250923114716.png]]
	- $\psi(x) \in \mathbb{R}^d$ is the output of the last hidden layer: this is the "embedding" of our user query $x$. 
	- $V_j \in \mathbb{R}^d$ is the vector of weights connecting the last hidden layer to the output for item $j$. This is the "embedding" of item $j$. 
	- The final prediction *is still a dot product*, but instead of $\langle U_i, V_j \rangle$ it's $\langle \psi(x), V_j \rangle$ 
	- We use cross-entropy loss with SGD to learn the model weights. 

Pro over traditional matrix factorization: 
- Traditional matrix factorization uses item-based user features, e.g. "user *i* watched item *j*"
- Deep factorization can decouple user features from specific items, e.g., user age, gender, behavioral patterns, preferences, context 
Con:
- We're not incorporating *item features* only *user features*

To solve this, we use a **two-tower neural network**: 
- One neural network maps user query features to query embedding: $x \rightarrow \psi(x) \in \mathbb{R}^d$ 
- One neural network maps item features to item embedding: $j \rightarrow \phi(j) \in \mathbb{R}^d$ 
- Then we take the dot product of $\langle \psi(x), \phi(j) \rangle$ as our output. 
	- Note that this isn't strictly a probability anymore. 

One common problem with deep factorization is **folding**: 
- If the model is only shown positive examples, the embeddings from different items/queries may end up in the same region, leading to spurious recommendations. E.g., items with different languages might end up together. 
- To address this, we can use **negative sampling** during training: we show the model negative samples to push them farther apart
	- We can also filter out such cases during candidate generation to avoid this during inference

# Retrieval (Candidate Generation)

- Can use KNN to find closest items in embedding space 
- For large-scale retrieval, this can be too slow
	- If the embedding is known statically, we can precompute list of top candidates for each query offline. Works well for related-item recommendations.
	- Use approximate nearest neighbors. 


# Scoring 

* Need to think very carefully about objective function for scoring: ![[Pasted image 20250923120940.png]]

# Re-Ranking
- Re-rank candidates according to score plus additional criteria, along with some filters:
	- Diversity
	- Filtering

# Factorization Machines

The above linear matrix factorization only models **2-way interactions:** between users and items. 

> "User_5" rated "Movie_10" 4.5

What if we wanted to model more complex interactions? 

> "User_5" rated "Movie_10" 4.5 and Genre=Action and Day=Weekend

The naive way to learn these weights would require exponential time! You'd need to learn:

> User x Movie, User x Time, User x Day, Movie x Time,  Movie x Day, ...

However, we can learn these weights in linear time. Any pairwise interaction can be computed as a dot product. 

> Interaction(User, Movie) = ⟨embedding_user, embedding_movie⟩
> Interaction(Time, Device) = ⟨embedding_Time, embedding_Device⟩

There is an algebra trick that lets us skip having to do all these pairwise interactions (which would be $\mathcal{O}(n^2)$) :

$$\sum_{i,j}{\langle e_i, e_j \rangle} = \text{sum of all embeddings}^2  - \frac{\text{sum of (embeddings squared)}}{2}$$ Factorization machines are **linear** not **deep**. But you can make it deep by feeding FM embeddings into a neural network. 