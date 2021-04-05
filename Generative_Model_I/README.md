# Laboratory 5

In this laboratory we will focus on generative models for classification. On this laboratory we will solve the **IRIS classification task using Gaussian classifiers**.



## Pre

The dataset is already available from ```sklearn library```

For this task we will use the function ```def load_iris ()```

Then we split the dataset into 100 samples for training and 50 samples for evaluation using the function ```def split_db_2to1 (D, L, seed = 0):```



## Multivariate Gaussian Classifier

The first model we implement is the Multivariate Gaussian Classifier (MVG).



The samples can be modeled as samples of multivariate Gaussian Distribuition with a class-dependent mean and covariance matrices.



The classifier has been implemented through a class with two main methods:



```def train(self, X, Y):```

The function compute the ML solution for the parameters is given by the empirical mean and covariance matrix of each class.



The solution is :



Compute the three likelihoods for all test samples.



For this task the prior probability is 1/3 for each class.



```def predict(self, X):```

Compute the join probabilities as:



Then compute the posterior probabilities for each class for each samples as:



 The predicted label is obtained as the class that has maximum posterior probability







