#!/usr/bin/python3
#-*- coding: utf-8 -*-
#
# Roll: 17NA30013
# Name: Navaneeth S
# Assignment no: 3
# Specific compilation/execution flags: python3 17NA30013_3.py
# Note: don't run this with python2

import os
import csv
import math
import pprint
import numpy as np
from numpy.random import choice

eps = np.finfo(float).eps

def node_entropy(data):
    entropy = 0.0
    values_arr = np.array(data['survived'])
    values, counts = np.unique(values_arr, return_counts=True)
    valcount = dict(zip(values, counts))
    for value in values:
        fraction = float(valcount[value])/len(values_arr)
        entropy += -fraction*(np.log2(fraction))
    return entropy

def entropy_attribute(data, attribute):
    att_arr = np.array(data[attribute])
    att_var, counts_att = np.unique(att_arr, True)
    target_arr = np.array(data['survived'])
    target_var, counts_tar = np.unique(target_arr, True)
    entropy2 = 0.0
    for variable in att_var:
        entropy = 0.0
        for target_variable in target_var:
            indx = []
            den = 0
            num = 0
            i = 0
            for att in data[attribute]:
                if att != variable:
                    pass
                else:
                    indx.append(i)
                i += 1
            for ind in indx:
                if data['survived'][ind] != target_variable:
                    pass
                else:
                    num += 1
            den = len(indx)
            fraction = float(num)/(den+eps)
            entropy += -fraction*np.log2(fraction+eps)
        fraction2 = float(den)/len(data['survived'])
        entropy2 += -fraction2*entropy
    return abs(entropy2)

def split_decide(data,feature):
    Entropy_att = []
    IG = [node_entropy(data) - entropy_attribute(data,key) for key in feature]
    prs = dict(zip(feature, IG))
    return max(prs, key=prs.get)

def predict(tree, example):
    for nodes in tree.keys():
        value = example[nodes]
        tree = tree[nodes][value]
        prediction = 0
        if isinstance(tree, dict):
            prediction = predict(tree, example)
        else:
            prediction = tree
            break
    return prediction

def subtree(data, value, node):
    cpy = {'age':[],
           'pclass':[],
           'survived':[],
           'gender':[]
           }
    indx = []
    i = 0
    for vals in data[node]:
        if vals != value:
            pass
        else:
            indx.append(i)
        i += 1
    for ind in indx:
        cpy['gender'].append(data['gender'][ind])
        cpy['age'].append(data['age'][ind])
        cpy['pclass'].append(data['pclass'][ind]) 
        cpy['survived'].append(data['survived'][ind])
    return cpy

def read_data(fname):
    with open(fname, 'r') as csvfile:
        rows=[]
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        fields=[]
        for row in csvreader:
            rows.append(row)
    data = {'pclass':[],'age':[],'gender':[],'survived':[]}
    for row in rows:
        data['survived'].append(row[3])
        data['gender'].append(row[2])
        data['pclass'].append(row[0])
        data['age'].append(row[1])
    return data

def create(data, node=None, feature=['pclass', 'age', 'gender'], tree=None):
    Class = list(data.keys())[-1]
    node = split_decide(data, feature)
    att_array = np.array(data[node])
    attValue = np.unique(att_array)
    newfeature = [feat for feat in feature if feat != node]
    if tree is None:
        tree = {}
        tree[node] = {}
    for value in attValue:
        subtable = subtree(data, value, node)
        cl_array = np.array(subtable['survived'])
        clvalue, counts = np.unique(cl_array, return_counts=True)
        if len(counts) == 1:
            tree[node][value] = clvalue[0]
        elif len(newfeature) == 0 or len(feature) == 0:
            tree[node][value] = max(data['survived'], key = data['survived'].count)
        else:
            tree[node][value] = create(subtable,node,newfeature)
    return tree

def ID3(data):
    tree = create(data)
    pprint.pprint(tree)
    return tree

def calculate_error(weights, tree, data):
    """Computes error and makes a list of indices of incorrectly classified"""
    incorrect_cls = []
    correct_cls = []
    error = 0
    for index in range(len(weights)):
        sample = {'age':data['age'][index],
                  'pclass': data['pclass'][index],
                  'gender': data['gender'][index]}
        result = predict(tree, sample)
        if result == 'no':
            result = 0
        elif result == 'yes':
            result = 1
        if not ((data['survived'][index] == 'no' and bool(result)==False) or
            (data['survived'][index] == 'yes' and bool(result)==True)):
            incorrect_cls.append(index)
            error = error + weights[index]
        else:
            correct_cls.append(index)
    return error, incorrect_cls, correct_cls

def sample_data(data, weights):
    sampled_data = {'age':[],
                    'pclass':[],
                    'survived':[],
                    'gender':[]
                    }
    age = []
    pclass = []
    survived = []
    gender = []
    N = len(data['pclass'])
    indicies = list(range(N))
    draw = choice(indicies, size=N, p=weights, replace=True)
    for index in draw:
        age.append(data['age'][index])
        survived.append(data['survived'][index])
        gender.append(data['gender'][index])
        pclass.append(data['pclass'][index])
    sampled_data['survived'] = survived
    sampled_data['gender'] = gender
    sampled_data['age'] = age
    sampled_data['pclass'] = pclass

    return draw, sampled_data

def trainandtestadaboost(times, traindata, testdata):
    """Takes in training and test data and trains the classifier"""
    #TRAINING
    alphas, classifiers = Adaboost(times, traindata)
    #TESTING
    N_test = len(testdata['pclass'])
    output = []

    for i in range(N_test):
        sample = {'gender': testdata['gender'][i],
                  'pclass': testdata['pclass'][i],
                  'age':testdata['age'][i],
                  }
        summation = 0
        for index in range(times): # Calculates based on sigma(alpha[i]*hypothesis(C[i]))
            try:
                predicted_cls = predict(classifiers[index], sample)
                if predicted_cls == 'yes':
                    summation += alphas[index] # C[i] = 1 for yes and 0 for no
                if not summation > 0:
                    output.append(0)
                else:
                    output.append(1)
            except:
                pass
    count = 0
    for i in range(N_test):
        if ((testdata['survived'][index] == 'yes' and bool(output[i])==True) or
            (testdata['survived'][index] == 'no' and bool(output[i])==False)):
            count += 1
    #accuracy = count/N_test
    return count/N_test

def update_weights(weights, alpha, incorrect_cls, correct_cls, sampled_indices):
    """updates weights based on the new alpha"""
    for j in correct_cls:
        weights[sampled_indices[j]] = weights[sampled_indices[j]] * ((math.e)**((-1)*(alpha))) 
    for i in incorrect_cls:
        weights[sampled_indices[i]] = weights[sampled_indices[i]] * ((math.e)**alpha)

    return [weight/sum(weights) for weight in weights]

def Adaboost(times, data):
    """Adaboost training algorithm. `times` is the number of times C acts."""
    N = len(data['pclass'])
    weights = []
    alphas = []
    classifiers = []
    weights += [1/N for _ in range(N)]
    for i in range(times):
        sampled_indices, sampled_data = sample_data(data, weights)
        print("Iteration:", i+1)
        classifiers.append(ID3(sampled_data))
        error, incorrect_cls, correct_cls = calculate_error(weights, ID3(sampled_data), sampled_data)
        try:
            alpha = 0.5 * math.log((1 - error)/error)
        except ZeroDivisionError:
            continue
        try:
            weights = update_weights(weights, alpha, incorrect_cls, correct_cls, sampled_indices)
        except:
            pass
        alphas.append(alpha)
    return alphas, classifiers

if __name__ == "__main__":
    traindata = read_data(os.path.join(os.getcwd(), 'data3_19.csv'))
    testdata = read_data(os.path.join(os.getcwd(), 'test3_19.csv'))
    accuracy = trainandtestadaboost(3, traindata, testdata)

    print(accuracy)
