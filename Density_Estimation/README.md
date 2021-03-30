# Laboratory 4

In this lab we will focus on calculating probability densities and ML estimates. The mathematical formulations implemented and the results obtained will be presented below.



## Gaussian density estimation

The purpose of this section is to estimate the parameters of a Gaussian distribution using the **maximum likelihood estimation**

#### GAU_pdf(x, mu, var) function 



![plot](https://github.com/luapicella/Machine-Learning-I/blob/main/Density_Estimation/image/Schermata%20da%202021-03-30%2019-51-59.png)

#### computeLikelihood(dataset, mu, var)



![Schermata da 2021-03-30 19-18-45](/image/Schermata da 2021-03-30 19-18-45.png)

#### GAU_logpdf(x, mu, v)

![Schermata da 2021-03-30 19-19-05](image/Schermata da 2021-03-30 19-19-05.png)

The above expression for the total probability is actually quite a pain to differentiate, so it is almost always simplified by taking the natural logarithm of the expression. This is absolutely fine because the natural logarithm is a monotonically increasing function. This means that if the value on the x-axis increases, the value on the y-axis also increases. This is important because it ensures that the maximum value of the log of the probability occurs at the same point as the original probability function. Therefore we can work with the simpler log-likelihood instead of the original likelihood.

#### computeLogLikelihood(dataset, mu, v)

![Schermata da 2021-03-30 19-19-22](./image/Schermata da 2021-03-30 19-19-22.png)

#### computeMaximumLikelihoodEstimates(dataset)

![Schermata da 2021-03-30 19-19-32](./image/Schermata da 2021-03-30 19-19-32.png)



### Result

In this case we have a good fit of the histogram, and the density well represents the distribution of ourdata. 



![Figure 2021-03-29 200935](./image/Figure 2021-03-29 200935.png)



Results with fewer samples are shown below. Respectively with 9000 samples, 5000 samples, 1000 samples and 100 samples.



![Figure 2021-03-30 193458](./image/Figure 2021-03-30 193458.png)



![Figure 2021-03-30 193505](./image/Figure 2021-03-30 193505.png)

![Figure 2021-03-30 193508](./image/Figure 2021-03-30 193508.png)

![Figure 2021-03-30 193512](./image/Figure 2021-03-30 193512.png)



## Multivariate Gaussian

In this section, The Multivarite Gaussian (MVG) will be implemented and the results will be compared with the data provided.

#### logpdf_GAU_ND(XND, mu, C) function

![Schermata da 2021-03-30 19-51-59](./image/Schermata da 2021-03-30 19-51-59.png)

where x is a feature vector.
