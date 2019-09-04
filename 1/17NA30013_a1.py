#!/usr/bin/python3
#-*- coding: utf-8 -*-
#
# Roll: 17NA30013
# Name: Navaneeth S
# Assignment no: 1
# Specific compilation/execution flags: python3 17NA30013_a1.py
# Note: don't run this with python2

import os
import numpy as np
from math import *

DATA_PATH = os.path.join(os.getcwd(), 'data1_19.csv')

def getm(data):
    n = 0
    p = 0
    for row in data:
        if row[-1] == 'no':
            n += 1
        else:
            p += 1
    return p > n

def init_entropy(data):
    n = 0
    p = 0
    for row in data:
        if row[-1] == 'no':
            n += 1
        else:
            p += 1

    s = 0
    if p and n:
        s = (-p*log(p/(p+n), 2) - n*log(n/(p+n), 2))/(p+n)
    return s

def compute_entropy(data, label, new):
    ans = np.zeros((len(new[label]), 2))
    total = len(data)
    s = 0
    for row in data:
        for i, header in enumerate(new[label]):
            if row[label] == header:
                if row[-1] == 'no':
                    ans[i][1] += 1
                else:
                    ans[i][0] += 1
                break
    for k in ans:
        sub_sum = k[0] + k[1]
        if k[0] and k[1]:
            s += (-k[0]*log(k[0]/sub_sum, 2) - k[1]*log(k[1]/sub_sum, 2))/total   
    return s

def build_tree(data, level, attr):
    first = init_entropy(data)
    max_label = 0
    max_sum = first - compute_entropy(data, 0, attr)
    if len(data[0]) < 1:
        return

    if first == 0 or len(data[0]) == 1:
        for j in range(level):
            print("--", end=" ")
        if getm(data) == 1:
            print(" yes")
        else:
            print(" no")
        return

    for k in range(1, len(data[0])-1):
        curr_sum = compute_entropy(data, k, attr)
        if first - curr_sum > max_sum:
            max_sum = first - curr_sum
            max_label = k

    if max_sum == 0:
        for j in range(level):
            print("--", end=" ")
        if getm(data) == 1:
            print(" yes")
        else:
            print (" no")
        return

    for k in range(len(attr[max_label])):
        data_new = []

        for j in range(level):
            print("--", end=" ")
        print(attr[max_label][k])
        for i in range(len(data)):
            if data[i][max_label] == attr[max_label][k]:
                data_new.append(data[i])
        data_new = np.delete(data_new,max_label,axis=1)
        new = attr.copy()
        del new[max_label]

        build_tree(data_new, level+1, new)
    return

def main():
    rawdata = open(DATA_PATH, 'r').read()
    data = [row.split(',') for row in rawdata.splitlines()[1:]]
    data = np.asarray(data)
    attr = [np.unique(data[:, 0]),
            np.unique(data[:, 1]),
            np.unique(data[:, 2])[::-1],
            np.unique(data[:, 3])[::-1]]
    build_tree(data, 0, attr)

if __name__ == '__main__':
    main()
