import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs


class MeanShift:
    ## Class Variables ##
    X = radius = bandwidth = max_itr = centroids = None

    ## Constructors ##
    def __init__(self, radius, bandwidth, max_itr=10):
        self.radius = radius
        self.bandwidth = bandwidth
        self.max_itr = max_itr

    ## Methods ##
    def parse_data(self, file_path, sep):
        df = pd.read_csv(file_path, sep=sep, header=None)
        self.X = df.values

    def gen_data(self):
        self.X, y = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=1)

    def clusterify(self):
        self._init_centroid()

        for i in range(self.max_itr):
            for j, centroid in enumerate(self.centroids):
                neighbours = self._neighbourhood_points(self.centroids[j])
                new_centroid = self._calc_new_centroid(self.centroids[j], neighbours)
                self.centroids[j] = new_centroid

    def _init_centroid(self):
        self.centroids = np.copy(self.X)

    def _neighbourhood_points(self, centroid):
        neighbours = []
        for i, old_centroid in enumerate(self.centroids):
            if self.euclidean_dist(old_centroid, centroid) <= self.radius:
                neighbours.append(old_centroid)

        return neighbours

    def _calc_new_centroid(self, centroid, neighbours):
        numer = denom = 0

        for neighbour in neighbours:
            weight = self._gaussian_kernel(self.euclidean_dist(neighbour, centroid))
            numer += (weight * neighbour)
            denom += weight

        return numer / denom

    def _gaussian_kernel(self, distance):
        return (1 / (self.bandwidth * np.sqrt(2 * np.pi))) \
               * np.exp(-0.5 * (distance / self.bandwidth) ** 2)

    def euclidean_dist(self, p1, p2):
        sqr_dist = 0

        for i in range(len(p1)):
            sqr_dist += (p1[i] - p2[i]) ** 2

        return np.sqrt(sqr_dist)


def main():
    mean_shift = MeanShift(radius=3, bandwidth=5)
    mean_shift.gen_data()
    mean_shift.clusterify()

    plt.scatter(mean_shift.X[:, 0], mean_shift.X[:, 1], s=30)

    for i, centroids in enumerate(mean_shift.centroids):
        plt.scatter(mean_shift.centroids[i][0], mean_shift.centroids[i][1], s=130, color='r', marker='x')

    plt.show()


if __name__ == "__main__":
    main()