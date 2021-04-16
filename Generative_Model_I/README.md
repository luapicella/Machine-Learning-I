# GENERATIVE MODEL I

In this laboratory we will focus on generative models for classification. 

# Introduction

On this laboratory we will solve the **IRIS classification task using Gaussian classifiers** and its variants.

In particular, the following classifiers were used:

- Multivariate Gaussian Classifier

- Naive Bayes Gaussian Classifier

- Tied Covariance Gaussian Classifier

  

## Pre

The dataset is already available from ```sklearn library```

For this task we will use the function ```def load_iris ()```

Then we split the dataset into 100 samples for training and 50 samples for evaluation using the function ```def split_db_2to1 (D, L, seed = 0):```



## Multivariate Gaussian Classifier

 ML solution for the parameters is given by the empirical mean and covariance matrix of each class:

![par](https://github.com/luapicella/Machine-Learning-I/blob/main/Generative_Model_I/Image/ML_solution.png)

## Naive Bayes Gaussian Classifier

If we know that, for each class, the different components are approximately independent, we can simplify the estimate. The ML solution is:

![par](https://github.com/luapicella/Machine-Learning-I/blob/main/Generative_Model_I/Image/Schermata%20da%202021-04-16%2023-49-19.png)

where the mean class matrix is the same and the covariance class matrix is a diagonal matrix.

## Tied Covariance Gaussian Classifier

If we assume that the covariance matrices of the different classes are tied, the ML solution is:

![par](https://github.com/luapicella/Machine-Learning-I/blob/main/Generative_Model_I/Image/Schermata%20da%202021-04-16%2023-54-28.png)

where the mean class matrix is the same and the covariance matrix is the within class covariance.



## Result

| Classifier               | Error rate |
| ------------------------ | ---------- |
| Multivariate Gaussian    | 4%         |
| Naive Bayes Gaussian     | 4%         |
| Tied Covariance Gaussian | <u>2%</u>  |











