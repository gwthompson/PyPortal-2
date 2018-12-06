from sklearn.decomposition import PCA
from PortalParams import Params, TrainResult
import utilities
import numpy as np
from mne.decoding import CSP

class Portal_CSP():
    def __init__(self, params, numCSP = 4):
        self.params = params
        self.trainResult = TrainResult()
        self.trainData = []
        self.labels= np.zeros((0))
        self.numCSP = numCSP

    def train(self,reader, doBalanceLabels):
        trainFiles = reader.traindata
        triggers = reader.classtrainTriggers
        onsets = reader.classtrainOnsets
        finOnsets = reader.classtrainFinOnsets
        [self.trainData, self.labels] = utilities.dataLoadFromEDF(self, trainFiles, triggers, onsets, finOnsets, self.params)
        if (doBalanceLabels):
            [self.trainData, self.labels] = utilities.balance_labels(self.trainData, self.labels)
        csp = CSP(n_components=self.numCSP, reg=None, log=True, norm_trace=False)
        csp.fit(self.trainData, self.labels)
        fVecs = csp.transform(self.trainData)
        self.trainResult.cspOp = csp
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
        #LDA!
        if not (self.params.finalClassifier is None):
            [Op, fTransformed] = utilities.trainClassifier(self.params.finalClassifier, fTransformed, labels)
            self.trainResult.finalOp = Op
        self.trainResult.trainTransformedVecs = fTransformed
        self.trainResult.trainLabels = labels

    def validate(self,reader):
        validateFiles = reader.testdata
        triggers = reader.classtestTriggers
        onsets = reader.classtestOnsets
        finOnsets = reader.classtestFinOnsets
        [validateData, validateLabels] = utilities.dataLoadFromEDF(self, validateFiles, triggers, onsets,finOnsets,self.params)
        fVecs = self.trainResult.cspOp.transform(validateData)
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
        #print('Class Rates:\n', classRates)
        #print('Confusion Matrix: \n', confMat)
        return classRates, confMat