from sklearn.decomposition import PCA
from PortalParams import Params, TrainResult
import utilities
import numpy as np
import SWCSP

class Portal_SWCSP():
    def __init__(self, params, numCSP):
        self.params = params
        self.trainResult = TrainResult()
        self.trainData = []
        self.labels= np.zeros((0))
        self.numCSP = numCSP

    def train(self,trainFiles,triggers,onsets,finOnsets, doBalanceLabels):
        [self.trainData, self.labels] = utilities.dataLoadFromEDF(self, trainFiles, triggers, onsets, finOnsets,self.params)
        self.numClasses = len(set(self.labels))
        if (doBalanceLabels):
            [self.trainData, self.labels] = utilities.balance_labels(self.trainData, self.labels)
        cspModels = []
        for i in range(self.numClasses):
            for j in range(self.numClasses):
                if i<j:
                    print("Training SWCSP Model for {} vs {}".format(i,j))
                    classFts1 = self.trainData[self.labels == i, :, :]
                    classFts2 = self.trainData[self.labels == j, :, :]
                    S = []
                    S.append(classFts1)
                    S.append(classFts2)
                    cspWorker = SWCSP.SWCSP(self.params.Fs, self.numCSP)
                    cspWorker.train(S)
                    cspModels.append(cspWorker)
        fVecs = np.zeros((self.labels.shape[0],self.numCSP*2*len(cspModels)))
        for i in range(self.labels.shape[0]):
            fvec=[]
            for j in range(len(cspModels)):
                fvec = np.concatenate((fvec, cspModels[j].process(self.trainData[i,:])))
            fVecs[i,:] = fvec

        self.trainResult.cspOp = cspModels
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
        # LDA!
        if not (self.params.finalClassifier is None):
            [Op, fTransformed] = utilities.trainClassifier(self.params.finalClassifier, fTransformed, labels)
            self.trainResult.finalOp = Op
        if not (self.params.distanceFun is None):
            self.trainResult.trainTransformedVecs = fTransformed
        self.trainResult.trainLabels = labels

    def validate(self,validateFiles,triggers,onsets,finOnsets):
        [validateData, validateLabels] = utilities.dataLoadFromEDF(self, validateFiles, triggers, onsets,finOnsets,self.params)
        fVecs = np.zeros((validateLabels.shape[0], self.numCSP * 2 * len(self.trainResult.cspOp)))
        for i in range(self.labels.shape[0]):
            fvec = []
            for j in range(len(self.trainResult.cspOp)):
                fvec = np.concatenate((fvec, self.trainResult.cspOp[j].process(validateData[i, :])))
            fVecs[i, :] = fvec
        #Norm!
        for i in range(fVecs.shape[0]):
            fVecs[i, :] = (fVecs[i, :] - self.trainResult.mean) / self.trainResult.std
        fTransformed = fVecs
        if (not self.params.finalClassifier is None):
            fTransformed = utilities.applyClassifier(self.params.finalClassifier, self.trainResult.finalOp, fTransformed)
        result = np.zeros((fTransformed.shape[0]))
        for i, fvec in enumerate(fTransformed):
            if not (self.params.distanceFun is None):
                result[i] = self.params.distanceFun(self.trainResult.trainTransformedVecs, self.trainResult.trainLabels, fvec, self.params)
            else:
                result[i] = fTransformed[i]
        classRates, confMat = utilities.calcStats(validateLabels,result)
        print('Class Rates:\n', classRates)
        print('Confusion Matrix: \n', confMat)