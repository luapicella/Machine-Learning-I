from utlis import load_iris, plot_hist, plot_scatter
import matplotlib.pyplot as plt



if __name__ == '__main__':
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load_iris('./data/iris.csv')
    plot_hist(D, L, "./images/withMean")
    plot_scatter(D, L, "./images/withMean")

    mu = D.mean(1)
    DC = D - mu.reshape((D.shape[0], 1))
    plot_hist(DC, L, "./images/withoutMean")
    plot_scatter(DC, L, "./images/withoutMean")
