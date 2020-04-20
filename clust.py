import numpy as np
import csv
import random
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
sns.set()

Y = []
X = []


def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))


def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum + 1, data_point, centroids[index_of_minimum]]


def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids) / 2


def iterate_k_means(data_points, centroids, total_iteration):
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)

    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point],
                                                                      centroids[index_centroid])
            label = assign_label_cluster(distance, data_points[index_point], centroids)
            centroids[label[0] - 1] = compute_new_centroids(label[1], centroids[label[0] - 1])

            if iteration == (total_iteration - 1):
                cluster_label.append(label)

    return [cluster_label, centroids]


def print_label_data(result):
    res = 0.0
    arr = []
    for data in result[0]:
        res += euclidean(result[1][data[0] - 1], data[1])
        arr.append(int(data[0]))
        print("cluster number: {} \n".format(data[0]))
    print("Last centroids position: \n {}".format(result[1]))
    print("WSS " + str(res))
    return arr, res

def jaccard(cluster, true_classes):
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for i in range(len(cluster) - 1):
        for j in range(i + 1, len(cluster)):
            if true_classes[i] == true_classes[j]:
                if cluster[i] == cluster[j]:
                    tp += 1
                else:
                    fp += 1
            else:
                if cluster[i] == cluster[j]:
                    tn += 1
                else:
                    fn += 1
    return tp / (tp + tn + fp)


def create_centroids(num):
    centroids = []
    for i in range(num):
        centroids.append([random.random(), random.random(), random.random()])
    return np.array(centroids)


if __name__ == "__main__":
    with open('data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if csv_reader.line_num != 1:
                attributes = [float(i) for i in row]
                Y.append(int(attributes.pop()))
                X.append(attributes)
    features = normalize(np.array(X))
    centroids = create_centroids(5)
    total_iteration = 100
    [cluster_label, new_centroids] = iterate_k_means(features, centroids, total_iteration)
    arr, WSS = print_label_data([cluster_label, new_centroids])

    pca = PCA(n_components=2)
    dataTwoDimensional = pca.fit_transform(features)
    x = []
    y = []
    for i in dataTwoDimensional:
        x.append(i[0])
        y.append(i[1])
    ax = sns.scatterplot(x=x, y=y, palette=['blue', 'red', 'green', 'black', 'yellow'], hue=Y)
    plt.show()
    plt.clf()
    ax = sns.scatterplot(x=x, y=y, hue=arr)
    plt.show()
    print(jaccard(arr, Y))
    w = []
    j = []
    for i in range(1, 10):
        features = normalize(np.array(X))
        centroids = create_centroids(i)
        total_iteration = 100
        [cluster_label, new_centroids] = iterate_k_means(features, centroids, total_iteration)
        arr, WSS = print_label_data([cluster_label, new_centroids])
        j.append(jaccard(arr, Y))
        w.append(WSS)
    print(w)
    print(j)
    plt.clf()
    sns.lineplot(x=[i for i in range(1, 10)], y=w)
    plt.show()
    plt.clf()
    sns.lineplot(x=[i for i in range(1, 10)], y=j)
    plt.show()
