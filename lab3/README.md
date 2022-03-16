# LAB 2 

### Dimensional reduction

- [PCA](https://github.com/luapicella/Machine-Learning-I/blob/main/lab3/pca.py)
- [LDA](https://github.com/luapicella/Machine-Learning-I/blob/main/lab3/lda.py)

### Result

We can observe that the first LDA direction (x-axis) results in lower overlapping of the green and the orange classes, which can thus be better separated along this direction. 

In contrast, we can observe that in LDA  classes are not well separated. This is because PCA is an unsupervised method and reduces the dimensionality by selecting the components with greater variance but does not take into account the classes. Instead LDA, which is a supervised method, tries to select the axes that best separate the classes.


| PCA             |  LDA |
:-------------------------:|:-------------------------:
![](https://github.com/luapicella/Machine-Learning-I/blob/main/lab3/image/PCA2.jpg?raw=true) | ![](https://github.com/luapicella/Machine-Learning-I/blob/main/lab3/image/LDA2.jpg?raw=true)
