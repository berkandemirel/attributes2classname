#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Attributes2Classname: A discriminative model for attribute-based unsupervised zero-shot learning.
# Written by berkan
# Contact: demirelberkan@gmail.com
# --------------------------------------------------------

import numpy as np
import tensorflow as tf
import scipy.io as sio
from scipy import spatial
import master, results
import tflearn, itertools
import pickle
from datetime import datetime

FLAGS = tf.app.flags.FLAGS


def extractData(data, dataName):

    dataContent = sio.loadmat(data)
    dataContent = dataContent[dataName]

    # Return feature matrix.
    return dataContent


def flatten(listOfLists):
    "Flatten one level of nesting"
    return itertools.chain.from_iterable(listOfLists)

def generateAverageWordVectors( wordVectors, vectorWeights):

    return vectorWeights.dot(wordVectors)

def generatePerturbedExamples( predicateMatrix, corruptionLevel ):

    newData = predicateMatrix
    for i in xrange(corruptionLevel-1):
        tmpPredicateMatrix = predicateMatrix
        r = np.random.random((len(predicateMatrix), len(predicateMatrix[0])))
        si = np.argsort(r)
        si = si[:, range(0,i)]

        for j in xrange(len(predicateMatrix)):
            tmpPredicateMatrix[j, si[j,:]] = np.logical_not(tmpPredicateMatrix[j, si[j,:]])
        newData = np.concatenate((newData, tmpPredicateMatrix), axis=0)

    return newData


def lossFunction( classVec, attributeVec, wrongClassVec, correctPredicateBasedAttrVec, wrongPredicateBasedAttrVec, hammingDistance ):

    classVec = classVec/tf.sqrt(tf.reduce_sum(tf.square(classVec), 1, keep_dims=True))
    attributeVec = attributeVec / tf.sqrt(tf.reduce_sum(tf.square(attributeVec), 1, keep_dims=True))
    correctPredicateBasedAttrVec = correctPredicateBasedAttrVec / tf.sqrt(tf.reduce_sum(tf.square(correctPredicateBasedAttrVec), 1, keep_dims=True))
    wrongPredicateBasedAttrVec = wrongPredicateBasedAttrVec / tf.sqrt(tf.reduce_sum(tf.square(wrongPredicateBasedAttrVec), 1, keep_dims=True))
    wrongClassVec = wrongClassVec / tf.sqrt(tf.reduce_sum(tf.square(wrongClassVec), 1, keep_dims=True))

    correctComb = tf.matmul(classVec, attributeVec, transpose_b=True)
    wrongComb =  tf.matmul(wrongClassVec, attributeVec, transpose_b=True)
    predicateBasedCorrectAttributeComb = tf.matmul(classVec, correctPredicateBasedAttrVec, transpose_b=True)
    predicateBasedWrongAttributeComb =  tf.matmul(classVec, wrongPredicateBasedAttrVec, transpose_b=True)

    if master.applyLossType == master.lossType[0]: #predicate matrix based
        return tf.maximum((predicateBasedWrongAttributeComb + hammingDistance) - predicateBasedCorrectAttributeComb, 0)
    elif master.applyLossType == master.lossType[1]: #image based
        return tf.maximum((wrongComb + hammingDistance) - correctComb, 0)
    else: #combined
        return tf.maximum((predicateBasedWrongAttributeComb + hammingDistance) - predicateBasedCorrectAttributeComb, 0) \
               + tf.maximum((wrongComb + hammingDistance) - correctComb, 0 )

def evalFunction( classVec, attributeVec, groundTruthLabels ):

    classVec = classVec/tf.sqrt(tf.reduce_sum(tf.square(classVec), 1, keep_dims=True))
    attributeVec = attributeVec / tf.sqrt(tf.reduce_sum(tf.square(attributeVec), 1, keep_dims=True))
    similarity = tf.matmul(classVec, attributeVec, transpose_b=True)

    return similarity

def batch_norm(x, n_out, phase_train, scope='bn'):

    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)

        batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def main(argv=None):

    with open('objs.pickle') as f:
        __C = pickle.load(f)

    # Get the data.
    train_classes_filename = __C.get('TRAIN_CLASS_PATH')
    test_classes_filename = __C.get('TEST_CLASS_PATH')
    attribute_vectors_filename = __C.get('ATTRIBUTE_VECTOR_PATH')
    predicate_matrix_filename = __C.get('PREDICATE_MATRIX_PATH')
    attr_classifiers_filename =  __C.get('ATTR_CLASSIFIER_RESULTS_PATH')
    groundtruth_labels_filename = __C.get('GROUND_TRUTH_LABELS')
    train_image_labels_filename = __C.get('TRAIN_IMAGE_LABELS')
    train_scores_filename = __C.get('TRAIN_SCORES')
    logFileName = __C.get('LOG_FILE')
    tmpFileName = __C.get('TMP_FILENAME')
    plotAccuracyPerNIter = __C.get('PLOT_ACC_PER_N_ITER')

    networkModel = __C.get('CURR_MODEL')

    # Get the number of epochs for training.
    num_epochs = __C.get('NUM_EPOCH')

    #Get the verbose status
    verbose = __C.get('VERBOSE')

    # Get the size of layer one.
    num_hidden = __C.get('CURR_HIDDEN')

    # Get the status of hand-crafted examples
    perturbed_examples = __C.get('PERTURBED_EXAMPLES')

    #Get the corruption level of hand-crafted examples
    corruption_level = __C.get('PERTURBED_EXAMPLE_CORRLEVEL')

    #get batch size
    batch_size = __C.get('MAX_BATCH_SIZE')-1

    trainClasses =  extractData(train_classes_filename, 'trainClasses')
    testClasses = extractData(test_classes_filename, 'testClasses')
    attributeVectors = extractData(attribute_vectors_filename, 'attributeVectors')
    predicateMatrix = extractData(predicate_matrix_filename, 'predicateMatrix')
    attributeClassifierResults = extractData(attr_classifiers_filename, 'attClassifierResults')
    groundTruthLabels = extractData(groundtruth_labels_filename, 'groundTruthLabels')
    trainImageLabels = extractData(train_image_labels_filename, 'trainImageLabels')
    trainScores = extractData(train_scores_filename, 'trainScores')

    # XXX TEMPORARY
    #trainClasses = trainClasses / np.linalg.norm(trainClasses, axis = 1, keepdims=True)
    #testClasses = testClasses / np.linalg.norm(testClasses, axis = 1, keepdims=True)
    #attributeVectors = attributeVectors / np.linalg.norm(attributeVectors, axis = 1, keepdims=True)

    # XXX TEMPORARY
    #const_scale=0.4
    #attributeVectors = attributeVectors*const_scale
    #trainClasses = trainClasses*const_scale
    #testClasses = testClasses*const_scale


    # Get the shape of the training data.
    train_size,num_features = trainClasses.shape

    # Get the shape of the training images.
    image_size, _ = predicateMatrix.shape

    # Get Average word vectors
    averageTrainAttributeVectors = generateAverageWordVectors( attributeVectors, trainScores )
    averageTrainPredicateMatrixBasedAttributeVectors = generateAverageWordVectors(attributeVectors, predicateMatrix)
    averageTestAttributeVectors = generateAverageWordVectors( attributeVectors, attributeClassifierResults )

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    classVecInput = tf.placeholder("float", shape=[None, num_features], name='CC')
    correctAttributeVecInput = tf.placeholder("float", shape=[None, num_features], name='CA')
    wrongPredicateBasedAttributeVecInput = tf.placeholder("float", shape=[None, num_features], name='WPA')
    correctPredicateBasedAttributeVecInput = tf.placeholder("float", shape=[None, num_features], name='CPA')
    hammingDistanceInput = tf.placeholder("float", shape=[None, None], name='HD')
    wrongClassVecInput = tf.placeholder("float", shape=[None, num_features], name='WC')
    groundTruthLabelsInput = tf.constant(groundTruthLabels.T, 'float')

    # hamming distance between class vectors.
    hammingDistClasses = np.zeros((len(predicateMatrix),len(predicateMatrix)), dtype=float)
    for i in xrange(len(predicateMatrix)):
        for j in xrange(len(predicateMatrix)):
            hammingDistClasses[i,j] = spatial.distance.hamming( predicateMatrix[i,:], predicateMatrix[j,:] )

    # Initialize the hidden weights and pass inputs
    with tf.variable_scope("wScope", reuse=False):
        wHidden = tf.get_variable('W1',
            shape=[num_features, num_hidden],
            initializer=tflearn.initializations.uniform_scaling(shape=None, factor=1.0, dtype=tf.float32, seed=0))

        wHidden2 = tf.get_variable('W2',
            shape=[num_hidden, num_hidden],
            initializer=tflearn.initializations.uniform_scaling(shape=None, factor=1.0, dtype=tf.float32, seed=0))

        firstLayer = tf.nn.tanh(tf.matmul(classVecInput, wHidden))
        correctClassOutput = tf.nn.sigmoid(tf.matmul(firstLayer, wHidden2))

    with tf.variable_scope("wScope", reuse=True):
        wHidden = tf.get_variable('W1')
        wHidden2 = tf.get_variable('W2')

        firstLayer = tf.nn.tanh(tf.matmul(correctAttributeVecInput, wHidden))
        correctAttributeOutput = tf.nn.sigmoid(tf.matmul(firstLayer, wHidden2))

    with tf.variable_scope("wScope", reuse=True):
        wHidden = tf.get_variable('W1')
        wHidden2 = tf.get_variable('W2')

        firstLayer = tf.nn.tanh(tf.matmul(wrongClassVecInput, wHidden))
        wrongClassOutput = tf.nn.sigmoid(tf.matmul(firstLayer, wHidden2))

    with tf.variable_scope("wScope", reuse=True):
        wHidden = tf.get_variable('W1')
        wHidden2 = tf.get_variable('W2')

        firstLayer = tf.nn.tanh(tf.matmul(correctPredicateBasedAttributeVecInput, wHidden))
        correctPredicateBasedAttributeOutput = tf.nn.sigmoid(tf.matmul(firstLayer, wHidden2))

    with tf.variable_scope("wScope", reuse=True):
        wHidden = tf.get_variable('W1')
        wHidden2 = tf.get_variable('W2')

        firstLayer = tf.nn.tanh(tf.matmul(wrongPredicateBasedAttributeVecInput, wHidden))
        wrongPredicateBasedAttributeOutput = tf.nn.sigmoid(tf.matmul(firstLayer, wHidden2))

    loss = tf.reduce_sum(
                lossFunction(correctClassOutput, correctAttributeOutput, wrongClassOutput,
                correctPredicateBasedAttributeOutput, wrongPredicateBasedAttributeOutput, hammingDistanceInput))

    # Optimization.
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

    accuracy =  evalFunction( correctClassOutput, correctAttributeOutput, groundTruthLabelsInput )

    classVectorsTensor = correctClassOutput
    attributeVectorsTensor = correctAttributeOutput

    #write results to the tmp file.
    file_ = open( tmpFileName, 'a' )

    logFile = open(logFileName, 'a')

    saver = tf.train.Saver()

    randomnessFlag = False

    timeStamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    loggedTrainData = []
    loggedTestData = []
    initializationFlag = False

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        try:
            if __C.get('SAVE_MODEL') == True:
                saver.restore(s, __C.get('LEARNED_MODEL_PATH')+str(num_hidden)+".ckpt")
            else:
                tf.initialize_all_variables().run()
        except:
            tf.initialize_all_variables().run()


        totalLoss = 0

        numberOfVectorPerIter = len( trainImageLabels )

        # Iterate and train.
        for step in xrange( num_epochs * image_size):

            offset = step % train_size

            currClassIndices = [i for i, x in enumerate(trainImageLabels) if x == offset+1] #is this class valid for training set?
            if currClassIndices != []:

                currTrainClass = trainClasses[offset:(offset + 1), :] # word vector of current training class

                # determine average word vector of attributes which is valid for currTraining Class
                currTrainAttributes = averageTrainAttributeVectors[currClassIndices, :]

                validIndices = range(0, numberOfVectorPerIter)
                validIndices = list(set(validIndices) - set(currClassIndices)) # find valid training indices for another classes
                invalidClasses = np.unique(trainImageLabels[validIndices]) # determine another classes
                wrongTrainClasses = trainClasses[invalidClasses-1, :] # word vectors of another classes

                currPredicateBasedTrainAttributes = averageTrainPredicateMatrixBasedAttributeVectors[np.unique(trainImageLabels[currClassIndices])-1,:]
                wrongPredicateBasedTrainAttributes = averageTrainPredicateMatrixBasedAttributeVectors[np.unique(invalidClasses-1,),:]


                if master.applyLossType == master.lossType[2]:
                    currPredicateBasedTrainAttributes = \
                        np.repeat(currPredicateBasedTrainAttributes, len(currTrainAttributes), axis=0)

                    repeatTimes = len(currTrainAttributes) / len(wrongPredicateBasedTrainAttributes)

                    wrongPredicateBasedTrainAttributes = \
                        np.repeat(wrongPredicateBasedTrainAttributes, repeatTimes+1, axis=0)

                    wrongPredicateBasedTrainAttributes = wrongPredicateBasedTrainAttributes[0:len(currTrainAttributes),:]

                currentHammingDistance = hammingDistClasses[offset:(offset + 1), invalidClasses-1]

                #forward pass
                _, curr_loss = s.run([train, loss], feed_dict={classVecInput: currTrainClass,
                                    correctAttributeVecInput: currTrainAttributes,
                                    wrongClassVecInput: wrongTrainClasses,
                                    correctPredicateBasedAttributeVecInput: currPredicateBasedTrainAttributes,
                                    wrongPredicateBasedAttributeVecInput: wrongPredicateBasedTrainAttributes,
                                    hammingDistanceInput: currentHammingDistance.T})

                totalLoss = curr_loss + totalLoss

                if offset == 0:
                    if verbose:
                        print 'Loss: ', totalLoss


                    trainAccuracy = 0
                    testAccuracy = 0

                    accuracyFlag = False

                    if (step % plotAccuracyPerNIter) == 0:
                        #evaluate network results
                        trainScores = \
                            accuracy.eval(feed_dict={classVecInput: trainClasses[np.unique(trainImageLabels)-1,:],
                                                     correctAttributeVecInput: averageTrainAttributeVectors})

                        trainAccuracy = results.getResults(trainImageLabels, trainScores)
                        print 'train Accuracy: ' + str(trainAccuracy)
                        accuracyFlag = True

                        testScores = \
                            accuracy.eval(feed_dict={classVecInput: testClasses,
                                                     correctAttributeVecInput: averageTestAttributeVectors})

                        testAccuracy = results.getResults(groundTruthLabels, testScores, False)
                        print 'Test Accuracy: ' + str(testAccuracy)


                    if initializationFlag == False:
                        if master.saveWordVectors == True:
                            initialTestClasses = \
                                classVectorsTensor.eval(feed_dict={classVecInput: testClasses,
                                                                   correctAttributeVecInput: averageTestAttributeVectors})
                            initialAttributes = \
                                attributeVectorsTensor.eval(feed_dict={classVecInput: testClasses,
                                                                       correctAttributeVecInput: averageTestAttributeVectors})
                            initialTrainClasses = \
                                classVectorsTensor.eval(feed_dict={classVecInput: trainClasses[np.unique(trainImageLabels) - 1, :],
                                               correctAttributeVecInput: averageTrainAttributeVectors})

                            initialTestScores = testScores

                        initializationFlag = True

                    if accuracyFlag == True:
                        loggedTrainData.append(trainAccuracy*100)
                        loggedTestData.append(testAccuracy*100)
                        logFile.write('#HiddenUnit:'+ str(__C.get('CURR_HIDDEN'))
                                      +',Step:'+str(step)+',Accuracy:'+str(testAccuracy*100) + '\n')

                        if master.applyCrossValidation == False:
                            results.drawAccuracyCurves(loggedTrainData, loggedTestData, timeStamp)

                    if (totalLoss <= __C.get('OVERFITTING_THRESHOLD') or __C.get('STOP_ITER') <= step) and step !=0:
                        testAccuracy = results.getResults(groundTruthLabels, testScores, False)
                        file_.write(str(testAccuracy) + '\n')
                        file_.close()
                        logFile.close()
                        results.getResults(groundTruthLabels, testScores, False, True)

                        if __C.get('SAVE_MODEL') == True:
                            saver.save(s, __C.get('LEARNED_MODEL_PATH')+str(num_hidden)+".ckpt")

                        if master.saveWordVectors == True:

                            wordVectorsSavePath = __C.get('WORD_VECTORS')

                            finalTrainClasses = \
                                classVectorsTensor.eval(feed_dict={classVecInput: trainClasses[np.unique(trainImageLabels) - 1, :],
                                               correctAttributeVecInput: averageTrainAttributeVectors})

                            finalTestClasses = \
                                classVectorsTensor.eval(feed_dict={classVecInput: testClasses,
                                                         correctAttributeVecInput: averageTestAttributeVectors})

                            finalAttributes = \
                                attributeVectorsTensor.eval(feed_dict={classVecInput: testClasses,
                                                         correctAttributeVecInput: averageTestAttributeVectors})

                            finalTestScores = testScores

                            sio.savemat(wordVectorsSavePath+'initialTestClasses.mat', {'initialTestClasses': initialTestClasses})
                            sio.savemat(wordVectorsSavePath+'finalTestClasses.mat', {'finalTestClasses': finalTestClasses})
                            sio.savemat(wordVectorsSavePath+'initialAttributes.mat', {'initialAttributes': initialAttributes})
                            sio.savemat(wordVectorsSavePath+'finalAttributes.mat', {'finalAttributes': finalAttributes})
                            sio.savemat(wordVectorsSavePath + 'initialTrainClasses.mat',{'initialTrainClasses': initialTrainClasses})
                            sio.savemat(wordVectorsSavePath + 'finalTrainClasses.mat',{'finalTrainClasses': finalTrainClasses})
                            sio.savemat(wordVectorsSavePath + 'initialTestScores.mat',{'initialTestScores': initialTestScores})
                            sio.savemat(wordVectorsSavePath + 'finalTestScores.mat',{'finalTestScores': finalTestScores})
                        return
                    totalLoss = 0

if __name__ == '__main__':
    tf.app.run()

