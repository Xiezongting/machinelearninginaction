# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:53:19 2017

@author: pcc

decesion tree
"""

from math import log
import operator

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
#   统计每个label出现的次数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
#   信息熵公式
    for key in labelCounts:
        prob =float(labelCounts[key])/numEntries
        shannonEnt -=prob*log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers'
    return dataSet,labels
        
def splitDataSet(dataSet, axis, value):
#   防止对原DataSet修改，故增加新的list
    retDataSet = []
#    extend增加列表的数字，appedn增加一个列表，具体可以百度
    for featVec in dataSet:
        if featVec[axis] ==value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature =-1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature =i
    return bestFeature

def majorityCnt(classList):
#    多数人投票
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] =0
        classCount[vote] +=1
#        字典排序,其他写法可百度
        sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
#    如果类别完全相同，则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
#    遍历完所有属性
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
        
def classify(inputTree,featLables,testVect):
    firstStr =inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLables.index(firstStr)
#    同样的，判断时候也是递归调用
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key].__name__=='dict'):
                classLabel = classify(secondDict[key],featLables,testtestVect)
            else :
                classLabel =secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr =open(filename)
    return pickle.load(fr)


    

    

