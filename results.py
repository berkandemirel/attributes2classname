#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Performance results of our ZSL approach
# Written by berkan
# Contact: demirelberkan@gmail.com
# --------------------------------------------------------

import numpy as np
import operator
import pickle, master
import matplotlib.pyplot as plt
from datetime import datetime

def drawAccuracyCurves( trainData, testData, timeStamp ):

    with open('objs.pickle') as f:
        __C = pickle.load(f)

    xLine = np.arange(len(trainData))*__C.get('PLOT_ACC_PER_N_ITER')

    fig = plt.figure()
    plt.plot(xLine, trainData)
    plt.plot(xLine, testData)

    plt.legend(['train accuracy', 'test accuracy'], loc='upper left')

    plt.savefig( __C.get('VISUAL_DATA')+'accuracyCurv'+'_'+str(__C.get('CURR_HIDDEN'))+'_'+timeStamp+'.pdf' )
    plt.close(fig)

def drawConfusionMatrix( confusionMatrix, timeStamp):

    with open('objs.pickle') as f:
        __C = pickle.load(f)

    if master.DATASET == master.datasetList[1]:#AwA dataset

        fig = plt.figure()
        plt.imshow(confusionMatrix, interpolation='nearest')
        plt.xticks(np.arange(0, 10),['pers. cat', 'hippo.', 'leopard', 'hump. whale', 'seal', 'chimpanzee', 'rat', 'g. panda','pig', 'raccoon'], rotation=60)
        plt.yticks(np.arange(0, 10),['pers. cat', 'hippo.', 'leopard', 'hump. whale', 'seal', 'chimpanzee', 'rat', 'g. panda','pig', 'raccoon'])
        plt.gcf().subplots_adjust(bottom=0.25)

        plt.savefig( __C.get('VISUAL_DATA')+'confMatrix_'+str(__C.get('CURR_HIDDEN'))+'_'+timeStamp+'.pdf' )
        plt.close(fig)

    elif master.DATASET == master.datasetList[0]:#aPaY dataset

        fig = plt.figure()
        plt.imshow(confusionMatrix, interpolation='nearest')
        plt.xticks(np.arange(0, 12),['bag', 'build.', 'carr.', 'cent.', 'donkey', 'goat', 'jetski', 'monk.','mug', 'statue', 'wolf', 'zebra'], rotation=60)
        plt.yticks(np.arange(0, 12),['bag', 'build.', 'carr.', 'cent.', 'donkey', 'goat', 'jetski', 'monk.','mug', 'statue', 'wolf', 'zebra'])
        plt.gcf().subplots_adjust(bottom=0.25)

        plt.savefig( __C.get('VISUAL_DATA')+'confMatrix_'+str(__C.get('CURR_HIDDEN'))+'_'+timeStamp+'.pdf' )
        plt.close(fig)

    else:
        pass

def drawBarChart(barList, timeStamp):

    width = .35

    with open('objs.pickle') as f:
        __C = pickle.load(f)

    if master.DATASET == master.datasetList[1]:  # AwA dataset

        fig = plt.figure()
        ind = np.arange(len(barList))
        plt.bar(ind, barList, width=width)
        plt.xticks(np.arange(0, 10),['pers. cat', 'hippo.', 'leopard', 'hump. whale', 'seal', 'chimpanzee', 'rat', 'g. panda','pig', 'raccoon'], rotation=60)

        plt.gcf().subplots_adjust(bottom=0.25)

        plt.savefig( __C.get('VISUAL_DATA')+'barChart_'+str(__C.get('CURR_HIDDEN'))+'_'+timeStamp+'.pdf' )
        plt.close(fig)
    elif master.DATASET == master.datasetList[0]:  #aPaY dataset

        fig = plt.figure()
        ind = np.arange(len(barList))
        plt.bar(ind, barList, width=width)
        plt.xticks(np.arange(0, 12),['bag', 'build.', 'carr.', 'cent.', 'donkey', 'goat', 'jetski', 'monk.','mug', 'statue', 'wolf', 'zebra'])

        plt.gcf().subplots_adjust(bottom=0.25)

        plt.savefig( __C.get('VISUAL_DATA')+'barChart_'+str(__C.get('CURR_HIDDEN'))+'_'+timeStamp+'.pdf' )
        plt.close(fig)

    else:
        pass


def getResults(groundTruthLabels, networkResults, detailedResult = False, drawResults = False):

    timeStamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    uniqueClasses = np.unique(groundTruthLabels.T)
    averageValue = 0

    confusionMatrix = [[0 for x in range(len(uniqueClasses))] for y in range(len(uniqueClasses))]
    barList = [0 for x in range(len(uniqueClasses))]

    for i in xrange(len(uniqueClasses)):
        indX = np.nonzero(groundTruthLabels.T == uniqueClasses[i])

        currValues = indX[1]
        currIndices = indX[1]
        counter = 0
        currList = []
        currClasses = []

        for j in indX[1]:
            index, value = max(enumerate(networkResults[:,j]), key=operator.itemgetter(1))
            confusionMatrix[i][index] = confusionMatrix[i][index] + 1
            currList.append(value)
            currClasses.append(index)
            currValues[counter] = value
            currIndices[counter] = uniqueClasses[index]
            counter = counter +1
        #x = sorted(range(len(currList)), key=lambda k: currList[k])
        #T = [i for i in x[1:5]]
        #print T
        correctIdx = np.nonzero(currIndices.T == uniqueClasses[i])
        averageValue = averageValue + float(len(correctIdx[0]))/len(indX[1])
        if detailedResult:
            print 'Accuracy for Class #'+str(i)+':'+str(float(len(correctIdx[0]))/len(indX[1]))

        barList[i] = float(len(correctIdx[0]))/len(indX[1])

    averageValue = averageValue/len(uniqueClasses)
    
    if drawResults==True and master.applyCrossValidation == False:
        drawConfusionMatrix(confusionMatrix, timeStamp)
        drawBarChart(barList, timeStamp)
    
    return averageValue

