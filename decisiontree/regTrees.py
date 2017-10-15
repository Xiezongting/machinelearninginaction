# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:47:09 2017

@author: pcc

CART decision tree
"""
import numpy as np

def loadDataSet(fileName):
    dataMat =[]
    fr = open(fileName)
#    map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，
#   并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
    for line in fr.readlines():
        curLine =line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
#    numpy的nonzero函数，具体百度.matrix a[:,x]表示取所有行，第x+1列
#   对于二维数组，nonzero函数返回两个array，第一个是行，第二个是列，取第一个
#   这样就可以选择dataSet中第feature列元素>value所处行
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:] #第一处错误修正  
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:] #第一处错误修正
    return mat0,mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])


def regErr(dataSet):
#    方差*个数，混乱程度
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType, ops=(1,4)):
#    tolS是容许的误差下降值,tolN是切分的最少样本数
    tolS = ops[0]
    tolN = ops[1]
#    所有label都相等
    if len(set(dataSet[:-1].T.tolist()[0])) ==1:
        return None, leafType(dataSet)
    m,n = np.shape(dataSet)
    S =errType(dataSet)
#    numpy中 inf表示无穷大
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split
    
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val =chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None :
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leatType, errType, ops)
    retTree['right'] = createTree(rSet, leatType, errType, ops)
    return retTree

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] =getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree, testData): 
    if shape(testData)[0] ==0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet =binSplitDataSet(testData, tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'],tree['spval'])
        errorNoMerge = np.sum(power(lSet[:,-1] - tree['left'],2)) + \
                        np.sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+ tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree
