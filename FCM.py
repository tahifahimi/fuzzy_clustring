import pandas as pd
import numpy as np
import random
import operator
import math
from matplotlib import pyplot as plt


def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x / summation for x in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat


def calculateClusterCenter(membership_mat):
    cluster_mem_val = [[], [], [], []]
    for s in range(n):
        cluster_mem_val[0].append(membership_mat[s][0])
        cluster_mem_val[1].append(membership_mat[s][1])
        cluster_mem_val[2].append(membership_mat[s][2])
        cluster_mem_val[3].append(membership_mat[s][3])

    cluster_centers = list()
    for j in range(k):
        # x = list(cluster_mem_val[j])
        x = cluster_mem_val[j]
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers


def updateMembershipValue(membership_mat, cluster_centers):
    p = float(2 / (m - 1))
    for i in range(n):
        x = list(df.iloc[i])
        # distances = [np.linalg.norm(map(operator.sub, x, cluster_centers[j])) for j in range(k)]
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j] / distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1 / den)
    return membership_mat


def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzyCMeansClustering():
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    while curr <= MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        curr += 1
    print(membership_mat)
    return cluster_labels, cluster_centers


def plot(labels):
    colors = ["r", "b", "g", "y"]
    for i in range(n):
        array = list(df.iloc[i])
        if labels[i] == 0:
            plt.scatter(array[0], array[1], color=colors[1])
        elif labels[i] == 1:
            plt.scatter(array[0], array[1], color=colors[0])
        elif labels[i] == 2:
            plt.scatter(array[0], array[1], color=colors[2])
        else:
            plt.scatter(array[0], array[1], color=colors[3])

    # plt.savefig('.png')
    plt.show()



if __name__=="__main__":
    # read file and initate the values
    df_full = pd.read_csv("sample2.csv")
    columns = list(df_full.columns)
    features = columns[:len(columns)]
    # print(features)
    class_labels = list(df_full[columns[-1]])
    df = df_full[features]
    print("the first coordinate", list(df.iloc[0]))
    # Number of Attributes
    attribute = len(df.columns)
    # Number of Clusters
    k = 4
    # Maximum number of iterations
    MAX_ITER = 10
    # Number of data points
    n = len(df)
    # Fuzzy parameter
    m = 2.00

    labels, centers = fuzzyCMeansClustering()
    print("the labels are", labels)
    plot(labels)
