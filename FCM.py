"""Tahere Fahimi 9539045
    fuzzy clustering method"""

import pandas as pd
import numpy as np
import random
import operator
import math
from matplotlib import pyplot as plt


# assign value randomly to the fuzzy value quantities
def initialize_fuzzy_quantities():
    matrix = list()
    for i in range(n):
        temp_list = [random.random() for i in range(c)]
        summation = sum(temp_list)
        # normalize the fuzzy quantities values
        matrix.append([x / summation for x in temp_list])
    return matrix


def find_Centers(u, c):
    cluster_quantity = []
    # create a list to save the fuzzy quantities for each center
    for i in range(c):
        cluster_quantity.append([])
    #  separate data in each cluster
    for s in range(n):
        for i in range(c):
            cluster_quantity[i].append(u[s][i])

    cluster_centers = list()
    for j in range(c):
        x = cluster_quantity[j]
        # calculate all u^m
        xraised = [e ** m for e in x]
        # Denominator of the division
        denominator = sum(xraised)
        temp_num = list()
        # numerator of the division
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        # divide the numerator / denominator
        center = [z / denominator for z in numerator]
        # add new center
        cluster_centers.append(center)
    return cluster_centers


def update_fuzzy_quantities(u, cluster_centers):
    power = float(2 / (m - 1))
    for i in range(n):
        x_y = list(df.iloc[i])
        # calculate distance of the data point from all centers ---> ||xk - vj||
        distances = [np.linalg.norm(list(map(operator.sub, x_y, cluster_centers[j]))) for j in range(c)]
        # ||xk - vi||/ ||xk-vj||
        for j in range(c):
            den = sum([math.pow(float(distances[j] / distances[k]), power) for k in range(c)])
            u[i][j] = float(1 / den)
    return u


# assign new cluster to each data point ---> find the biggest fuzzy
# quantities for each node and assign that center to that data
def change_clusters(u):
    cluster_labels = list()
    for i in range(n):
        max_val, index = max((val, idx) for (idx, val) in enumerate(u[i]))
        cluster_labels.append(index)
    return cluster_labels


def fcm(c):
    # u matrix => the matrix of the fuzzy quantities
    u = initialize_fuzzy_quantities()
    iteration_number = 0
    while iteration_number <= maximum_iteration:
        # find new centers
        centers = find_Centers(u, c)
        # calculate fuzzy quantities based on new centers
        u = update_fuzzy_quantities(u, centers)
        # assign new center to each data point
        labels = change_clusters(u)
        iteration_number += 1
    return labels


# draw the final vision of the coordinates
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


if __name__ == "__main__":
    # read file and initiate the values
    df_full = pd.read_csv("sample3.csv")
    columns = list(df_full.columns)
    features = columns[:len(columns)]
    # print(features)
    class_labels = list(df_full[columns[-1]])
    df = df_full[features]
    print("the first coordinate", list(df.iloc[0]))
    # Number of Clusters
    c = 2
    # Maximum number of iterations
    maximum_iteration = 10
    # Number of data points
    n = len(df)
    # Fuzzy parameter
    m = 1.80

    labels = fcm(c)
    print("the labels are", labels)
    plot(labels)
