# Why is it helpful? 

- Some algorithms, e.g., linear/logistic regression, are sensitive to collinear features. 
- For certain models it can speed up training up time and help with interpretability to have fewer features. Also reduces storage and memory need. 
- Can reduce curse of dimensionality and prevent overfitting
- Less feature engineering and overhead required

# Algorithms
- Univariate tests, e.g., variance thresholding (remove low-variance features)
- Bivariate tests, e.g., correlation
- Forward/Backward selection
- LASSO
- Elastic net - LASSO + ridge