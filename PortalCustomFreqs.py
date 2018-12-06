from sklearn.decomposition import PCA
from PortalParams import Params, TrainResult
import utilities
import numpy as np
import scipy.io
from scipy.fftpack import fftfreq
import os
import math

class Portal_CustomFreqs():
    #Portal: CustomFreqs version
    def __init__(self, params):
        self.params = params
        self.trainResult = TrainResult()
        self.trainData = []
        self.labels= np.zeros((0))

    def train(self,reader,doBalanceLabels):

        trainFiles = reader.traindata
        triggers = reader.classtrainTriggers
        onsets = reader.classtrainOnsets
        finOnsets = reader.classtrainFinOnsets

        [self.trainData, self.labels] = utilities.dataLoadFromEDF(self, trainFiles, triggers, onsets, finOnsets,self.params)

        if (doBalanceLabels):
            [self.trainData, self.labels] = utilities.balance_labels(self.trainData, self.labels)

        numChannels = self.trainData.shape[1]
        numSamples = self.trainData.shape[0]
        fvecLen = len(self.params.channelSelect)
        fVecs = np.zeros((numSamples,fvecLen))
        for k in range(numSamples):
            #print("Fvec: " + str(k) + " of " + str(numSamples))
            for i in range(fvecLen):
                sample = np.squeeze(self.trainData[k, i, :])
                if not (self.params.fDiaps is None):
                    self.params.lowFreq = self.params.fDiaps[i][0]
                    self.params.highFreq = self.params.fDiaps[i][1]
                specVal = np.mean(utilities.get1DFeatures(sample, self.params)) #!!!!!!!
                fVecs[k, i] = specVal
        #Shuffle!
        inds = np.random.permutation(fVecs.shape[0])
        fVecs = fVecs[inds,:]
        labels = self.labels[inds]
        self.trainResult.mean = np.mean(fVecs,0)
        self.trainResult.std = np.std(fVecs,0)
        #Norm!

        for i in range(fVecs.shape[0]):
            fVecs[i,:] = (fVecs[i,:]-self.trainResult.mean)/self.trainResult.std
        fTransformed = fVecs
        # PCA!
        if self.params.usePCA:
            pcaTransform = PCA(self.params.numPC)
            pcaTransform.fit(fVecs)
            self.trainResult.pcaOp = pcaTransform
            fTransformed = pcaTransform.transform(fVecs)
        #LDA!
        if not (self.params.finalClassifier is None):
            [Op, fTransformed] = utilities.trainClassifier(self.params.finalClassifier, fTransformed, labels)
            self.trainResult.finalOp = Op
        if not (self.params.distanceFun is None):
            self.trainResult.trainTransformedVecs = fTransformed
        self.trainResult.trainLabels = labels

    def validate(self,reader):
        validateFiles = reader.testdata
        triggers = reader.classtestTriggers
        onsets = reader.classtestOnsets
        finOnsets = reader.classtestFinOnsets
        [validateData, validateLabels] = utilities.dataLoadFromEDF(self, validateFiles, triggers, onsets,finOnsets,self.params)
        numChannels = validateData.shape[1]
        numSamples = validateData.shape[0]
        fvecLen = len(self.params.channelSelect)
        fVecs = np.zeros((numSamples, fvecLen))
        for k in range(numSamples):
            #print("Fvec: " + str(k) + " of " + str(numSamples))
            for i in range(fvecLen):
                sample = np.squeeze(validateData[k, i, :])
                if not (self.params.fDiaps is None):
                    self.params.lowFreq = self.params.fDiaps[i][0]
                    self.params.highFreq = self.params.fDiaps[i][1]
                specVal = np.mean(utilities.get1DFeatures(sample, self.params))
                fVecs[k, i] = specVal
        #Norm!
        for i in range(fVecs.shape[0]):
            fVecs[i, :] = (fVecs[i, :] - self.trainResult.mean) / self.trainResult.std
        fTransformed = fVecs
        if self.params.usePCA:
            fTransformed = self.trainResult.pcaOp.transform(fVecs)
        if (not self.params.finalClassifier is None):
            fTransformed = utilities.applyClassifier(self.params.finalClassifier, self.trainResult.finalOp, fTransformed)
        result = np.zeros((fTransformed.shape[0]))
        for i, fvec in enumerate(fTransformed):
            if not (self.params.distanceFun is None):
                result[i] = self.params.distanceFun(self.trainResult.trainTransformedVecs, self.trainResult.trainLabels, fvec, self.params)
            else:
                result[i] = fTransformed[i]
        classRates, confMat = utilities.calcStats(validateLabels,result)
        #print('Class Rates:\n', classRates)
        #print('Confusion Matrix: \n', confMat)
        return classRates, confMat