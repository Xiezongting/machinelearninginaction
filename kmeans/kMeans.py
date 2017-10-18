# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:38:16 2017

@author: xzt

K-means算法
"""
import numpy as np

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
#    计算欧式距离
    return np.sqrt(sum(np.power(vecA - vecB,2)))

def randCent(dataSet, k):
#    随机选取质心
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j] - minJ))
        centroids[:,j] = minJ +rangeJ * np.random.rand(k,1)
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
#    第一列记录簇索引值，第二列存储误差
    clusterAssment =np.mat(np.zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged =True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf 
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):
#           取出属于同一簇的元素
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust,axis=0)
    return centroids,clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
#    第一列记录簇索引值，第二列存储误差
    clusterAssment =np.mat(np.zeros((m,2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
#    全体数据的质心作为第一个
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
    while(len(centList) < k):
        lowestSSE = np.inf
        for i in range(len(centList)):
#            在第i个簇中的数据集,尝试2分，比较是否合适
            ptsInCurrClust = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
#            2个质点，以及分割后的距离(第一列标记属于0还是1)
            centroidMat, splitClustAss = kMeans(ptsInCurrClust, 2,distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit =sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit,and notSplit "+str(sseSplit)+" , "+str(sseNotSplit))
            if(sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
#       确定完最优分割点后，开始替换值
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print("the bestCentToSplit is %d:"%bestCentToSplit)
        print("the len of bestClustAss is %d:"%len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return np.mat(centList),clusterAssment

datMat = np.mat(loadDataSet("testSet.txt"))
centList, myNewAssments = biKmeans(datMat,3)
        
            
                