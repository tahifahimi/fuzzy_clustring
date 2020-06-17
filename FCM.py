"""Tahere Fahimi 9539045
    fuzzy clustering method"""

import pandas as pd
import numpy as np
import random
import operator
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

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

    # calculate the Error
    error = 0.0
    for k in range(n):
        x_y = list(df.iloc[k])
        # calculate distance of the data point from all centers ---> ||xk - vj||
        distances = [np.linalg.norm(list(map(operator.sub, x_y, centers[i])))**2 for i in range(c)]
        for i in range(c):
            error += distances[i]*u[k][i]

    return labels, error, centers


# draw the final vision of the coordinates
def plot(labels, centers):
    ax = plt.figure().add_subplot(111)
    colors = ["r", "b", "g", "y", "c", "m", "y", "k", "gray"]
    for i in range(n):
        array = list(df.iloc[i])
        if labels[i] == 0:
            ax.scatter(array[0], array[1], color=colors[0])
        elif labels[i] == 1:
            ax.scatter(array[0], array[1], color=colors[1])
        elif labels[i] == 2:
            ax.scatter(array[0], array[1], color=colors[2])
        elif labels[i] == 3:
            ax.scatter(array[0], array[1], color=colors[3])
        elif labels[i] == 4:
            ax.scatter(array[0], array[1], color=colors[4])
        elif labels[i] == 5:
            ax.scatter(array[0], array[1], color=colors[5])
        elif labels[i] == 6:
            ax.scatter(array[0], array[1], color=colors[6])
        else:
            ax.scatter(array[0], array[1], color=colors[7])
    # Add rectangles
    width = 0.01
    height = 0.01
    for x in centers:
        a_x, a_y = x[0], x[1]
        ax.add_patch(
            Rectangle(xy=(a_x - width / 2, a_y - height / 2), width=width, height=height, linewidth=1, color='blue',
                      fill=False))
    ax.axis('equal')
    plt.savefig(file_name+"clusterNo"+str(c)+'.png')
    # plt.show()


if __name__ == "__main__":
    # read file and initiate the values
    file_name = "sample4.csv"
    df_full = pd.read_csv(file_name)
    columns = list(df_full.columns)
    features = columns[:len(columns)]
    # print(features)
    class_labels = list(df_full[columns[-1]])
    df = df_full[features]
    print("the first coordinate", list(df.iloc[0]))
    # Maximum number of iterations
    maximum_iteration = 1000
    # Number of data points
    n = len(df)
    # Fuzzy parameter
    m = 1.2

    # find the best clustering number from 2 - 8
    for i in range(2, 9):
        # Number of Clusters
        c = i
        labels, error, centers = fcm(c)
        print(error)
        plot(labels, centers)
