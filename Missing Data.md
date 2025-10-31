
# Types of missingness:
* **Missing Not at Random:** reasons why a value is missing is because of the value itself. E.g. people with higher incomes tend not to self-report their income. 
* **Missing at Random:** reasons why a value is missing is due to another variable. E.g., males with higher income tend not to self-report their income. 
* **Missing Completely at Random:** there is no pattern to the missingness. In practice, this is very rare. 

# Methods to deal with missing values
* **Deletion:** either column deletion or row deletion. Simple but deletion can destroy valuable information or create biases in your model. E.g., removing all rows with missing incomes can remove all rows with gender=male.
* **Imputation:** 
	* Mean/mode imputation for continuous data
	* Default value, e.g., empty string or -1 for categorical/ordinal data
	* KNN imputation
* **Surrogate splits** for tree-based models

# Case Studies
- How to deal with features with high (> 30%) missingness? 
	- First, determine type of missingness
	- If MCAR, we can impute 
	- If MAR, we can impute based on other variables
	- If MNAR, often using "missing" category is best approach