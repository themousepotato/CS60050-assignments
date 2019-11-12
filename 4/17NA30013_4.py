#!/usr/bin/python3
#-*- coding: utf-8 -*-
#
# Roll: 17NA30013
# Name: Navaneeth S
# Assignment no: 4
# Specific compilation/execution flags: python3 17NA30013_3.py
# Note: don't run this with python2

import numpy as np
import csv

DATASET = 'data4_19.csv'
k = 3
iterations = 10

def convert(data_points):
    data = []
    for points in data_points:
        temp = []        
        for ind, val in enumerate(points):
            if ind >= 4:
                temp.append(val)
            else:
                temp.append(float(val))

        data.append(temp)
    return data

def csv_data(filename):
    with open(filename,'rt') as csv_file:
        contents = csv.reader(csv_file)
        return [row for row in contents][:-1]  

def assignment(data,centroids):
    clusters = [[] for i in range(k)]
    for point in data:
        min_val = 10**9
        min_ind = None
        for i in range(k):
            if point_dist(point, centroids[i]) < min_val:
                min_val = point_dist(point, centroids[i])
                min_ind = i

        clusters[min_ind].append(point)

    return clusters

def calc_jaccard(A,B):
    intersection = 0
    for ele_A in A:
        if ele_A not in B:
            pass
        else:
            intersection += 1
    
    return 1 - intersection/(len(A) + len(B) - intersection)

def initialize(data):
    indices = np.random.choice(len(data),3,replace=False)
    return [data[val] for val in indices]

def point_dist(a,b):
    return np.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2 + (a[3] - b[3])**2 )

def update(clusters):
    centroids = []
    cluster_conv = [[] for i in range(k)]
    for i in range(k):
        cluster_conv[i] = [point[:-1] for point in clusters[i]]
        centroids.append(np.mean(cluster_conv[i],axis=0))
    
    return centroids

def jaccard(clusters, data):
    ground_truth = {
        keey : []
        for keey in np.unique(np.array(data)[:,4])
    }

    for val in data:
        ground_truth[val[4]].append(val)

    for i in range(k):
        print()
        for flower in ground_truth:
            print("Jaccard Distance between Cluster {} and Cluster {}: {:.3}".format(i+1,flower,calc_jaccard(ground_truth[flower],clusters[i])))

def main():
    data_points = csv_data(DATASET)
    data = convert(data_points)
    centroids = initialize(data)
    for i in range(iterations):
        clusters = assignment(data, centroids)
        centroids = update(clusters)
    
    print("\nAfter 10 iterations, the means of the clusters are:")
    for i in range(k):
        print("Cluster {}: {:.3}, {:.3}, {:.3}, {:.3}".format(i+1,centroids[i][0],centroids[i][1],centroids[i][2],centroids[i][3]))
    
    jaccard(clusters, data)

if __name__ == "__main__":
    main()