#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Master File
# Written by berkan
# Contact: demirelberkan@gmail.com
# --------------------------------------------------------

from easydict import EasyDict as edict
import subprocess, os
import pickle
import operator

__C = edict()
cfg = __C

tmpFileName = 'tmpfile'

datasetList = {0:"data/aPaY", 1:"data/AwA", 2:"data/CUB"}
sideInformationList = {0: "GloVe/AWV", 1: "GloVe/FWV", 2: "word2vec"}
networkModelList = {0: "VGG19", 1: "GoogleNet"}
lossType = {0: "predicateBased", 1: "imageBased", 2: "combined"}

DATASET = datasetList[1]
languageModel = sideInformationList[0]
networkModel = networkModelList[0]
applyLossType = lossType[1]
applyCrossValidation = False
saveWordVectors = False

def prepareData(mainPath, stopIter = 0, overfittingThreshold = 0):

    __C = edict()

    if DATASET == datasetList[0]:
        __C.SAVE_FILE = 'aPaYResults.txt'
        __C.VISUAL_DATA = 'visual/aPaY/'
        __C.LEARNED_MODEL_PATH = 'models/aPaY/'
        __C.WORD_VECTORS = 'wordVectors/aPaY/'
        __C.LOG_FILE = 'aPaYLog.txt'

    elif DATASET == datasetList[1]:
        __C.SAVE_FILE = 'AwAResults.txt'
        __C.VISUAL_DATA = 'visual/AwA/'
        __C.WORD_VECTORS = 'wordVectors/AwA/'
        __C.LEARNED_MODEL_PATH = 'models/AwA/'
        __C.LOG_FILE = 'AwALog.txt'
    elif DATASET == datasetList[2]:
        __C.SAVE_FILE = 'CUBResults.txt'
        __C.VISUAL_DATA = 'visual/CUB/'
        __C.LEARNED_MODEL_PATH = 'models/CUB/'
        __C.WORD_VECTORS = 'wordVectors/CUB/'
        __C.LOG_FILE = 'CUBLog.txt'

    else:
        pass

    if applyCrossValidation == False:
        __C.SAVE_MODEL = True
    else:
        __C.SAVE_MODEL = False

    __C.NUM_EPOCH = 1000000
    __C.NUM_HIDDEN = [100]#[100, 200, 300, 400, 500]
    __C.VERBOSE = True
    __C.PERTURBED_EXAMPLES = False
    __C.PERTURBED_EXAMPLE_CORRLEVEL = 5
    __C.MAX_BATCH_SIZE = 64
    __C.NUMBER_OF_FOLD = 2
    __C.PLOT_ACC_PER_N_ITER = 100
    __C.OVERFITTING_THRESHOLD = overfittingThreshold
    __C.CV_PATH = DATASET + '/cv_data/'
    __C.TMP_FILENAME = tmpFileName

    if stopIter == 0:
        __C.STOP_ITER = __C.NUM_EPOCH
    else:
        __C.STOP_ITER = stopIter

    __C.TRAIN_CLASS_PATH = mainPath +'/'+languageModel+ '/trainClasses.mat'
    __C.TEST_CLASS_PATH = mainPath +'/'+languageModel+ '/testClasses.mat'
    __C.ATTRIBUTE_VECTOR_PATH = mainPath +'/'+languageModel+ '/attributeVectors.mat'
    __C.PREDICATE_MATRIX_PATH = mainPath + '/predicateMatrix.mat'
    __C.ATTR_CLASSIFIER_RESULTS_PATH = mainPath +'/'+networkModel+ '/attClassifierResults.mat'
    __C.GROUND_TRUTH_LABELS = mainPath + '/groundTruthLabels.mat'
    __C.TRAIN_IMAGE_LABELS = mainPath + '/trainImageLabels.mat'
    __C.TRAIN_SCORES = mainPath +'/'+networkModel+'/trainScores.mat'

    return __C

cfg = __C


if __name__ == '__main__':

    try:
        os.remove(os.getcwd() + '/' + tmpFileName)
    except:
        pass

    if applyCrossValidation:

        __C = prepareData(DATASET,  30000,0)

        try:
            os.remove(os.getcwd() + '/' + __C.LOG_FILE)
        except:
            pass

        for currHidden in __C.NUM_HIDDEN:

            for currFold in xrange(__C.NUMBER_OF_FOLD):

                __C = prepareData(__C.CV_PATH+str(currFold+1), 30000,0)
                __C.CURR_HIDDEN = currHidden

                with open('objs.pickle', 'w') as f:
                    pickle.dump(__C, f)

                subprocess.call('python zsl.py', shell=True)

        with open(__C.LOG_FILE, "r") as ins:
            cvAccr = {}
            for line in ins:
                currentAccuracy = float(line.split(',')[2].split(':')[1])
                currentKey = line.split(',')[0]+' '+line.split(',')[1]
                if currentKey in cvAccr:
                    if cvAccr[currentKey] == 0:
                        cvAccr[currentKey] = currentAccuracy
                    else:
                        cvAccr[currentKey] = (cvAccr[currentKey] + currentAccuracy)/2
                else:
                    cvAccr[currentKey] = currentAccuracy

            print max(cvAccr.iteritems(), key=operator.itemgetter(1))[0]

    else:
        __C = prepareData(DATASET, 26000, 0)

        try:
            os.remove(os.getcwd() + '/' + __C.LOG_FILE)
        except:
            pass


        for currHidden in __C.NUM_HIDDEN:

            __C.CURR_HIDDEN = currHidden

            with open('objs.pickle', 'w') as f:
                pickle.dump(__C, f)

            subprocess.call('python zsl.py', shell=True)

            file = open(__C.SAVE_FILE, 'a')

            resultList = [float(line.rstrip('\n')) for line in open(__C.TMP_FILENAME)]

            os.remove(os.getcwd() + '/' + __C.TMP_FILENAME)

            file.write('ACCURACY: '+ str(resultList)+'\n')
            file.write('NUM_HIDDEN: '+ str(currHidden)+'\n')
            file.write('###############\n')
            file.close()
