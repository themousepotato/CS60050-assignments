#!/usr/bin/python3
#-*- coding: utf-8 -*-
#
# Roll: 17NA30013
# Name: Navaneeth S
# Assignment no: 3
# Specific compilation/execution flags: python3 17NA30013_3.py
# Note: don't run this with python2

import csv
import math
import numpy as np
import os
import pprint

eps = np.finfo(float).eps

def node_entropy(data):
    entropy = 0.0
    values_arr = np.array(data['survived'])
    values,counts = np.unique(values_arr,return_counts=True)
    valcount = dict(zip(values,counts))
    for value in values:
        fraction = float(valcount[value])/len(values_arr)
        entropy += -fraction*(np.log2(fraction))
    return entropy

def entropy_attribute(data, attribute):
    target_arr = np.array(data['survived'])
    target_var,counts_tar = np.unique(target_arr, True)
    att_arr = np.array(data[attribute])
    att_var, counts_att = np.unique(att_arr, True)
    entropy2 = 0.0
    for variable in att_var:
        entropy=0.0
        for target_variable in target_var:
            num = 0
            i=0
            den = 0
            indx = []
            for att in data[attribute]:
                if att == variable:
                    indx.append(i)
                i+=1
            for ind in indx:
                if data['survived'][ind] == target_variable:
                    num+=1
            den = len(indx)
            fraction = float(num)/(den+eps)
            entropy += -fraction*np.log2(fraction+eps)
        fraction2 = float(den)/len(data['survived'])
        entropy2 += -fraction2*entropy
    return abs(entropy2)

def subtree(data, value, node):
    cpy = {'pclass':[],'age':[],'gender':[],'survived':[]}
    indx = []
    i = 0
    for vals in data[node]:
        if vals == value:
            indx.append(i)
        i = i+1
    for ind in indx:
        cpy['pclass'].append(data['pclass'][ind])
        cpy['age'].append(data['age'][ind])
        cpy['gender'].append(data['gender'][ind])
        cpy['survived'].append(data['survived'][ind])
    return cpy

def split_decide(data,feature):
    IG = []
    Entropy_att = []
    for key in feature:
        IG.append(node_entropy(data) - entropy_attribute(data,key))
    prs = dict(zip(feature,IG))
    return max(prs, key = prs.get)

def create(data, node=None, feature=['pclass','age','gender'], tree=None):
    Class = list(data.keys())[-1]
    node = split_decide(data, feature)
    newfeature = [feat for feat in feature if feat != node]
    att_array = np.array(data[node])
    attValue = np.unique(att_array)
    if tree is None:
        tree = {}
        tree[node] = {}
    for value in attValue:
        subtable = subtree(data, value, node)
        cl_array = np.array(subtable['survived'])
        clvalue, counts = np.unique(cl_array, return_counts=True)
        if len(feature)==0 or len(newfeature)==0:
            tree[node][value] = max(data['survived'], key = data['survived'].count)
        elif len(counts)==1:
            tree[node][value] = clvalue[0]
        else:
            tree[node][value] = create(subtable,node,newfeature)
    return tree

def predict(tree, example):
    for nodes in tree.keys():
        prediction = 0
        value = example[nodes]
        tree = tree[nodes][value]
        if type(tree) is dict:
            prediction = predict(tree, example)
        else:
            prediction = tree
            break
    return prediction

def ID3(data):
    tree = create(data)
    pprint.pprint(tree)
    return tree

def read_data(fname):
    with open(fname, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields=[]
        fields = next(csvreader)
        rows=[]
        for row in csvreader:
            rows.append(row)
    data = {'pclass':[],'age':[],'gender':[],'survived':[]}
    for row in rows:
        data['pclass'].append(row[0])
        data['age'].append(row[1])
        data['gender'].append(row[2])
        data['survived'].append(row[3])
    return data

# Adaboost
def sample_data(data, weights):
    sampled_data = {'pclass':[],'age':[],'gender':[],'survived':[]}
    pclass = []
    age = []
    gender = []
    survived = []
    N = len(data['pclass'])
    indicies = list(range(N))
    from numpy.random import choice
    draw = choice(indicies, N, p=weights)
    for index in draw:
        pclass.append(data['pclass'][index])
        age.append(data['age'][index])
        gender.append(data['gender'][index])
        survived.append(data['survived'][index])
    sampled_data['pclass'] = pclass
    sampled_data['age'] = age
    sampled_data['gender'] = gender
    sampled_data['survived'] = survived
    return draw, sampled_data

def calculate_error(weights, tree, data):
    """Computes error and makes a list of indices of incorrectly classified"""
    error = 0
    incorrect_cls = []
    correct_cls = []
    for index in range(len(weights)):
        sample = {'pclass': data['pclass'][index] , 'age':data['age'][index],
                  'gender':data['gender'][index]}
        result = predict(tree, sample)
        if result == "yes":
            result = 1
        else:
            result = 0
        if ((data['survived'][index] == "no" and bool(result)==False) or
            (data['survived'][index] == "yes" and bool(result)==True)):
            correct_cls.append(index)
        else:
            error = error + weights[index]
            incorrect_cls.append(index)
    return error , incorrect_cls, correct_cls

def compute_alpha(error):
    """Computes alpha based on error"""
    alpha = 0.5 * math.log((1 - error)/error)
    return alpha

def Adaboost(times, data):
    """Adaboost training algorithm. `times` is the number of times C acts."""
    weights = []
    classifiers = []
    alphas = []
    N = len(data['pclass'])
    for _ in range(N):
        weights.append(1/N)
    for _ in range(times):
        sampled_indices, sampled_data = sample_data(data, weights)
        tree = ID3(sampled_data)
        classifiers.append(tree)
        error, incorrect_cls, correct_cls = calculate_error(weights, tree, sampled_data)
        alpha = compute_alpha(error)
        alphas.append(alpha)
        weights = update_weights(weights, alpha, incorrect_cls, correct_cls, sampled_indices)
    return alphas, classifiers 

def update_weights(weights, alpha, incorrect_cls, correct_cls, sampled_indices):
    """updates weights based on the new alpha"""
    for i in incorrect_cls:
        weights[sampled_indices[i]] = weights[sampled_indices[i]] * ((math.e)**alpha)
    for j in correct_cls:
        weights[sampled_indices[j]] = weights[sampled_indices[j]] * ((math.e)**((-1)*(alpha))) 
    n_factor = sum(weights)#Normalization Factor sigma(D'[index])
    for k in range(len(weights)):
        weights[k] = weights[k]/n_factor
    return weights

def trainandtestadaboost(times, traindata, testdata):
    """Takes in training and test data and trains the classifier"""
    # Training
    alphas, classifiers = Adaboost(times, traindata)

    #Testing
    output = []
    class1 = 0
    class2 = 0
    N_test = len(testdata['pclass'])

    for i in range(N_test):
        result = 0
        sample = {'pclass': testdata['pclass'][i] , 'age':testdata['age'][i],
                  'gender': testdata['gender'][i]}
        for index in range(times):#Calculates based on sigma(alpha[i]*hypothesis(C[i]))
            predicted_cls =predict(classifiers[index], sample)
            if predicted_cls == "yes":
                class1 = class1 + alphas[index] 
            else:
                class2 = class2 + alphas[index]
        if (class1>=class2):
            output.append(1)
        else:
            output.append(0)

    count = 0
    for i in range(N_test):
        if ((testdata['survived'][index] == "no" and bool(output[i])==False) or
            (testdata['survived'][index] == "yes" and bool(output[i])==True)):
            count = count + 1
    accuracy = count/N_test
    return accuracy

def main():
    trainset = os.path.join(os.getcwd(), 'data3_19.csv')
    traindata = read_data(trainset)
    testset = os.path.join(os.getcwd(), 'test3_19.csv')
    testdata = read_data(testset)
    accuracy = trainandtestadaboost(3, traindata, testdata)
    print('Accuracy: ', accuracy)

if __name__ == '__main__':
    main()