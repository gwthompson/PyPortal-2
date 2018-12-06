from sklearn.decomposition import PCA
from PortalParams import Params, TrainResult
import utilities
import numpy as np

class PortalIndepChan():
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
        [self.trainData, self.labels] = utilities.dataLoadFromEDF(self, trainFiles, triggers, onsets, finOnsets, self.params)
        if (doBalanceLabels):
            [self.trainData, self.labels] = utilities.balance_labels(self.trainData, self.labels)
        numChannels = self.trainData.shape[1]
        numSamples = self.trainData.shape[0]
        tmpVec = utilities.get1DFeatures(np.squeeze(self.trainData[0, 0, :]), self.params)
        fvecLen = len(tmpVec)
        fVecs = np.zeros((numSamples,numChannels,fvecLen))
        for k in range(numSamples):
            #print("Fvec: " + str(k) + " of " + str(numSamples))
            for i in range(numChannels):
                fVecs[k, i, :] = utilities.get1DFeatures(np.squeeze(self.trainData[k, i, :]), self.params)
        #Shuffle!
        inds = np.random.permutation(fVecs.shape[0])
        fVecs = fVecs[inds,:,:]
        labels = self.labels[inds]

        #Norm!
        for i in range(fVecs.shape[1]):
            chanVecs = np.squeeze(fVecs[:, i, :])
            self.trainResult.mean.append(np.mean(chanVecs, 0))
            self.trainResult.std.append(np.std(chanVecs, 0))
            for k in range(chanVecs.shape[0]):
                chanVecs[k, :] = (chanVecs[k, :] - self.trainResult.mean[i]) / self.trainResult.std[i]
            fVecs[:,i,:] = chanVecs
        fTransformed = np.zeros((fVecs.shape[0],fVecs.shape[1]*fVecs.shape[2]))
        if (self.params.usePCA):
            fTransformed = np.zeros((fVecs.shape[0],fVecs.shape[1]*self.params.numPC))
        # PCA or reshaping:
        for i in range(numChannels):
            chanVecs = np.squeeze(fVecs[:,i,:])
            if (self.params.usePCA):
                curPcaTransform = PCA(self.params.numPC)
                curPcaTransform.fit(chanVecs)
                self.trainResult.pcaOp.append(curPcaTransform)
                fTransformed[:,i*self.params.numPC:(i+1)*self.params.numPC] = curPcaTransform.transform(chanVecs)
            else:
                fTransformed[:, i * fVecs.shape[2]:(i + 1) * fVecs.shape[2]] = chanVecs
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
        [validateData, validateLabels] = utilities.dataLoadFromEDF(self, validateFiles, triggers, onsets,finOnsets, self.params)
        numChannels = validateData.shape[1]
        numSamples = validateData.shape[0]
        tmpVec = utilities.get1DFeatures(np.squeeze(validateData[0, 0, :]), self.params)
        fvecLen = len(tmpVec)
        fVecs = np.zeros((numSamples, numChannels, fvecLen))
        for k in range(numSamples):
            #print("Fvec: " + str(k) + " of " + str(numSamples))
            for i in range(numChannels):
                fVecs[k, i, :] = utilities.get1DFeatures(np.squeeze(validateData[k, i, :]), self.params)
        #Norm!
        fTransformed = np.zeros((fVecs.shape[0], fVecs.shape[1] * fVecs.shape[2]))
        if (self.params.usePCA):
            fTransformed = np.zeros((fVecs.shape[0], fVecs.shape[1] * self.params.numPC))
        for i in range(numChannels):
            chanVecs = np.squeeze(fVecs[:,i,:])
            for k in range(chanVecs.shape[0]):
                chanVecs[k, :] = (chanVecs[k, :] - self.trainResult.mean[i]) / self.trainResult.std[i]
            if (self.params.usePCA):
                fTransformed[:, i * self.params.numPC:(i + 1) * self.params.numPC] = self.trainResult.pcaOp[i].transform(chanVecs)
            else:
                fTransformed[:, i * fVecs.shape[2]:(i + 1) * fVecs.shape[2]] = chanVecs
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


