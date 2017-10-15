# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:49:14 2017

@author: xzt

KNN算法
"""

import numpy as np
import operator
import os

def createDataSet():
#    带有四个点的简单例子
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


#   inX是输入向量
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
#    numpy.tile(A,reps) 简单理解是此函数将A进行重复输出
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances =sqDiffMat.sum(axis=1)
#    很明显用的是欧氏距离
    distances =sqDistances**0.5
#    indicies 原来是Index的复数哈哈
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
#        取值，取不到默认为0
        classCount[voteILabel] = classCount.get(voteILabel,0) +1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename) :
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0;
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVector

#==============================================================================
# import matplotlib.pyplot as plt
# datingDataMat,datingLabels =file2matrix("datingTestSet2.txt")
# fig = plt.figure()
# plt.scatter(datingDataMat[:,1],datingDataMat[:,2],
#             15.0*np.array(datingLabels),15.0*np.array(datingLabels))
# plt.show()
#==============================================================================

def autoNorm(dataSet):
#    newValue = （oldValue - min)/(max-min)
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges =maxVals -minVals
    normDataSet =np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio =0.10
    datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount =0.0
    for i in range(numTestVecs):
        classifierResult =classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with"+str(classifierResult)+", the real answer is "+ str(datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is %f" %float(errorCount/numTestVecs))

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
#   加载训练集
    trainingFileList = os.listdir('trainingDigits')          
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
#    构造矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
#   进行判断
    testFileList = os.listdir('testDigits')       
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest))) 
