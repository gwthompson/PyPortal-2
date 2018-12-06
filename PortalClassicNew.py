from sklearn.decomposition import PCA
from PortalParams import Params, TrainResult
import utilities
import numpy as np

class Portal_Classic_New():
    def __init__(self, params, customFreq):
        self.params = params
        self.customFreq = customFreq
        self.trainResult = TrainResult()
        self.trainData = []
        self.labels= np.zeros((0))

    def train(self,reader, doBalanceLabels):
        trainFiles = reader.traindata
        triggers = reader.classtrainTriggers
        onsets = reader.classtrainOnsets
        finOnsets = reader.classtrainFinOnsets
        [self.trainData, self.labels] = utilities.dataLoadFromEDF(self, trainFiles, triggers, onsets, finOnsets, self.params)
        if (doBalanceLabels):
            [self.trainData, self.labels] = utilities.balance_labels(self.trainData, self.labels)
        numChannels = self.trainData.shape[1]
        numSamples = self.trainData.shape[0]
        sumFvecLen = 0
        for ch in range(numChannels):
            if not (self.params.fDiaps is None):
                self.params.lowFreq = self.params.fDiaps[ch][0]
                self.params.highFreq = self.params.fDiaps[ch][1]
            tmpVec = utilities.get1DFeatures(np.squeeze(self.trainData[0, ch, :]), self.params)
            sumFvecLen = sumFvecLen + len(tmpVec)
        fVecs = np.zeros((numSamples,sumFvecLen))
        for k in range(numSamples):
            #print('Train Sample ', k, ' of ', numSamples)
            curFvec = np.zeros((0))
            for i in range(numChannels):
                if not (self.params.fDiaps is None):
                    self.params.lowFreq = self.params.fDiaps[i][0]
                    self.params.highFreq = self.params.fDiaps[i][1]
                chVec = utilities.get1DFeatures(np.squeeze(self.trainData[k, i, :]), self.params)
                curFvec = np.concatenate((curFvec,chVec),axis=0)
            fVecs[k, :] = curFvec
        #Shuffle!
        inds = np.random.permutation(fVecs.shape[0])
        fVecs = fVecs[inds,:]
        labels = self.labels[inds]
        self.trainResult.mean = np.mean(fVecs,0)
        self.trainResult.std = np.std(fVecs,0)
        #Norm!

        for i in range(fVecs.shape[0]):
            fVecs[i,:] = (fVecs[i,:]-self.trainResult.mean)/self.trainResult.std
        # PCA!
        fTransformed = fVecs
        if self.params.usePCA:
            pcaTransform = PCA(self.params.numPC)
            pcaTransform.fit(fVecs)
            self.trainResult.pcaOp = pcaTransform
            fTransformed = pcaTransform.transform(fVecs)
        #LDA!
        if not (self.params.finalClassifier is None):
            [Op, fTransformed] = utilities.trainClassifier(self.params.finalClassifier, fTransformed, labels)
            self.trainResult.finalOp = Op
        self.trainResult.trainLabels = labels
        if not (self.params.distanceFun is None):
            self.trainResult.trainTransformedVecs = fTransformed

    def validate(self,reader):
        validateFiles = reader.testdata
        triggers = reader.classtestTriggers
        onsets = reader.classtestOnsets
        finOnsets = reader.classtestFinOnsets
        [validateData, validateLabels] = utilities.dataLoadFromEDF(self, validateFiles, triggers, onsets,finOnsets, self.params)
        numChannels = validateData.shape[1]
        numSamples = validateData.shape[0]

        if self.customFreqs:
            fvecLen = len(self.params.channelSelect)  # cust
            fVecs = np.zeros((numSamples, fvecLen))
            for k in range(numSamples):  # cust
                # print("Fvec: " + str(k) + " of " + str(numSamples))
                for i in range(fvecLen):
                    sample = np.squeeze(validateData[k, i, :])
                    if not (self.params.fDiaps is None):
                        self.params.lowFreq = self.params.fDiaps[i][0]
                        self.params.highFreq = self.params.fDiaps[i][1]
                    specVal = np.mean(utilities.get1DFeatures(sample, self.params))
                    fVecs[k, i] = specVal
        else:
            sumFvecLen = 0 #class
            for i in range(numChannels):
                if not (self.params.fDiaps is None):
                    self.params.lowFreq = self.params.fDiaps[i][0]
                    self.params.highFreq = self.params.fDiaps[i][1]
                tmpVec = utilities.get1DFeatures(np.squeeze(self.trainData[0, i, :]), self.params)
                sumFvecLen = sumFvecLen + len(tmpVec)
            fVecs = np.zeros((numSamples, sumFvecLen))

            for k in range(numSamples): #class
            #print('Validate Sample ', k, ' of ', numSamples)
                curFvec = np.zeros((0))
                for i in range(numChannels):
                    if not (self.params.fDiaps is None):
                        self.params.lowFreq = self.params.fDiaps[i][0]
                        self.params.highFreq = self.params.fDiaps[i][1]
                    chVec =  utilities.get1DFeatures(np.squeeze(validateData[k, i, :]), self.params)
                    curFvec = np.concatenate((curFvec,chVec),axis=0)
                fVecs[k,:] = curFvec

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

    def processChunk(self, rawTest):
        testChunk = utilities.preprocessDataChunk(rawTest, self)
        numChannels = testChunk.shape[0]
        curFvec = np.array([])
        for ch in range(numChannels):
            if not (self.params.fDiaps is None):
                self.params.lowFreq = self.params.fDiaps[ch][0]
                self.params.highFreq = self.params.fDiaps[ch][1]
            chVec = utilities.get1DFeatures(np.squeeze(self.trainData[0, ch, :]), self.params)
            curFvec = np.concatenate((curFvec, chVec))
        # Norm!
        curFvec = (curFvec - self.trainResult.mean) / self.trainResult.std
        fTransformed = np.expand_dims(curFvec,0)
        if self.params.usePCA:
            fTransformed = self.trainResult.pcaOp.transform(fTransformed)
        if (not self.params.finalClassifier is None):
            fTransformed = utilities.applyClassifier(self.params.finalClassifier, self.trainResult.finalOp,
                                                     fTransformed)
        if not (self.params.distanceFun is None):
            result = self.params.distanceFun(self.trainResult.trainTransformedVecs, self.trainResult.trainLabels, fTransformed, self.params)
        else:
            result = fTransformed
        return result

