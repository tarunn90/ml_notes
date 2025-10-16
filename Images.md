# Preprocessing

- **Resizing:** most models will expect images of a certain dimension, e.g., 224x224. 
- **Scaling:** you will probably need to scale the pixel values whether you are training your own model or using pretrained (e.g., CLIP)
	- **Min-max scaling:** divide by 255 so the range is $[0,1]$ 
	- **Standardization:** z-score standardization => $\mathcal{N}(0,1)$ 

# Representation Learning
How do we learn useful image embeddings? We need to pick a task which learns embeddings with good generalization. 

Methods include:
- Contrastive Learning
