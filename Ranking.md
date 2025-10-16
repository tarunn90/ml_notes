## Objective 
Problem statement: given a user input image, we want to return images that are the most similar to that input. 

# Clarifying Questions
- What media do we support? Text, images, video? 
- Are the query results personalized to the user? 
- What kind of latency is expected? Near real-time? 
## Representation Learning
Given some kind of unstructured data - image or text - we want to transform that into an embedding which can then be compared against other embeddings. Note the similarity with Rec Sys! 

## Labels
What are our labels here? Imagine you are trying to show images similar to the one given by the image. The labels could be:
- **Implicit**: User clicks
- **Explicit:** User likes or shares

# Model Development
For representation learning, we pick a task which gives us generalizable embeddings, e.g., contrastive learning.

Note that we can use a **pre-trained model** and fine-tune it. This has the benefits of a) speeding up training b) requiring less data. 
## Contrastive Learning
In contrastive learning, we pick dissimilar images and the model learns embeddings such that dissimilar images are pushed apart in the embedding space. 

To be specific, during training we have the following:
- A query image $q$, e.g., German Shepherd
- A positive image $p$, e.g., Belgian Malinois
- $n-1$ negative images, e.g., cats or horses. 

How do we label images as positive or negative?
- Human judgment 
	- Accurate but slow and expensive
- Interactions, i.e., if user queried image A and clicked on B then they are similar
	- Fast and cheap but noisy
- Self-supervision: create a similar image from the given image using augmentation, e.g., rotation

## Contrastive Loss
Before we start, we need to find a good similarity metric:
- Dot product/Cosine similarity: cheap and simple
- Euclidean distance: fails in high dimensions due to curse of dimensionality

To compute the loss for a single query image, we do the following:
1. Compute the similarities between the query image $q$ and the other images: if the other images are $r_1, ... r_n$ then we compute $s_1, ..., s_n$ 
2. We pass the similarity scores through a softmax to convert them into probabilities: $p_1, ..., p_n$ 
3. Finally we pass all of those into a cross-entropy layer to get the overall loss: 
	   $\text{Cross Entropy Loss} = -\sum_{i=1}^n y_i \log(p_i)$  
		where $y_i$ is the label ${0,1}$ for the correct image 


# Offline Evaluation
## Binary Metrics 
These metrics work well when the relevance can be summarized in a binary label. 
- **Recall@k:** what % of all relevant items are in the top $k$ items returned by our model?
	- Not helpful since usually there are *many* relevant items compared to $k$ so recall will always be low 
- **Precision@k**: for the top $k$ items returned by our model, what % are relevant to the query? $$\text{precision@k} = \frac{\text{number of relevant items among top $k$}}{k}$$
	- Not ideal since it doesn't give us the ranking quality *within* $k$ 
- **Mean Average Precision @ k, mAP@k:** for each output list, we average the precision, and then we average those values across the dataset. $$\begin{aligned}
  AP@k &= \frac{1}{R}\sum_{i=1}^k \text{precision}(i) \times \text{rel}(i) \\
mAP &= \frac{1}{Q} \sum_{q=1}^Q AP(q)
  \end{aligned}$$
  where:
  - $R$ = total number of relevant items
  - $\text{rel}(i)$ = $\mathbb{1}(\text{item i is relevant})$ 
  - $Q$ = number of queries

## Continuous Metrics
- DCG and nDCG

# Online Evaluation
- Watch Time
- Click-Through Rate

# Serving
## Indexing Service
As each image enters the platform, it gets indexed for use in embedding generation and nearest neighbor. 

## Embedding Generation Service
Using our trained Contrastive Learning model, we generate embeddings for each image:
- Preprocess image
- Pulls trained model from registry
- Runs inference
- Stores embeddings

## Nearest Neighbor Service
Given query $q$ and set of images $S$, we want to retrieve the closest images to $q$ in $S$. 

### Exact Nearest Neighbor
This is simply KNN. The time complexity is $O(N \times D)$ where $N$ is the number of images in our index and $D$ is the dimensionality of the embeddings. Since this is linear time and we may have billions of images, it may not be feasible in production. 

### Approximate Nearest Neighbor (ANN)
We use data structures to return faster but non-exact results. 
- **Tree-Based ANN:** split the space into partitions such that, for a binary tree, the search is $O(\log N \times D)$. The results for a query image are all images in the same leaf node as the query. 
- **Locality Sensitive Hashing (LSH):** we use a hash function to put the images into buckets. Then the query results are all images in the same bucket as the query. Lookup then becomes $O(D)$. 
- **Hierarchical Navigable Small worlds (HNSW):** Build a multi-layer graph where the upper layer has fewer points and is like the "highway" and the bottom layer has more points and is like the "surface roads". Starting with the query $q$, you compare $q$ to the neighbors of the entry node, find the closest neighbor, and continue until you hit the bottom layer and return the $k$ nearest neighbors. Lookup becomes $O(\log n)$. 

## Re-Ranking
We apply business logic to filter out unwanted images, e.g., private images, duplicates, etc. 

## Hard Negatives
Useful concept in any ML domain where distinguishing similar items is critical, e.g., facial recognition, retrieval, search, recommendation. The concept is to find cases the model got wrong and explicitly feed those into the model for training: this helps the model learn more nuanced representations. 

Example:
- User query is an image of a *golden retriever puppy playing in grass* 
- Positive results would be other golden retriever puppies playing in the grass 
- **Easy negatives** would be cars in driveways, kids playing soccer, etc. 
- **Hard negatives** would be similar but different images: adult golden retrievers, yellow lab puppies playing in grass, etc. 
- Without hard negatives, the model might learn "fuzzy + outdoor = match". 

