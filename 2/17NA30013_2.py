#!/usr/bin/python3
#-*- coding: utf-8 -*-
#
# Roll: 17NA30013
# Name: Navaneeth S
# Assignment no: 2
# Specific compilation/execution flags: python3 17NA30013_2.py
# Note: don't run this with python2

import os
import pandas as pd

TRAINING_PATH = os.path.join(os.getcwd(), 'data2_19.csv')
TESTING_PATH = os.path.join(os.getcwd(), 'test2_19.csv')

def get_df():
	df_train = pd.read_csv(TRAINING_PATH, delimiter=',')
	df_test = pd.read_csv(TESTING_PATH, delimiter=',')

	for df in [df_train, df_test]:
		for i in range(len(df.columns[0].split(','))):
			df[df.columns[0].split(',')[i]]=df[df.columns[0]].apply(lambda x: x.split(',')[i])

	df_train = df_train.drop(['D,X1,X2,X3,X4,X5,X6'], axis=1)
	df_test = df_test.drop(['D,X1,X2,X3,X4,X5,X6'], axis=1)
	return df_train, df_test

def classify(x):
    probab = probabs(x)
    mx = 0
    mxcl = ''
    for cl,pr in probab.items():
        if pr > mx:
            mx = pr
            mxcl = cl
    return mxcl

def probabs(x):
    probab = {}
    for cl in classes:
        pr = probcl[cl]
        for col,val in x.iteritems():
            try:
                pr *= probs[cl][col][val]
            except KeyError:
                pr = 0
        probab[cl] = pr
    return probab

def main():
	df, test = get_df()

	global probs, probcl, classes

	target = 'D'
	features = df.columns[df.columns != target]
	classes = df[target].unique()

	probs = {}
	probcl = {}
	for x in classes:
		mushcl = df[df[target]==x][features]
		clsp = {}
		tot = len(mushcl)
		for col in mushcl.columns:
			colp = {}
			for val,cnt in mushcl[col].value_counts().iteritems():
				pr = (cnt)/(tot)
				colp[val] = pr
			clsp[col] = colp
		probs[x] = clsp
		probcl[x] = (len(mushcl))/(len(df))

	b = []
	for i in df.index:
		b.append(classify(df.loc[i,features]) == df.loc[i,target])
	print('Training Accuracy:', sum(b)/len(df))

	b = []
	for i in test.index:
		b.append(classify(test.loc[i,features]) == test.loc[i,target])
	print('Test Accuracy:',sum(b)/len(test))

if __name__ == '__main__':
	main()