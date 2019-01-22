from sklearn.decomposition import PCA
from PortalParams import Params, TrainResult
import utilities
import numpy as np


class Portal_Classic_New2():
    def __init__(self, params, mode='Classic'): #modes: 'Classic', 'CustomFreqs', 'IndepChan'
        self.params = params
        self.mode = mode
        self.trainResult = TrainResult()
        self.trainData = []
        self.labels = np.zeros((0))



    def train(self, reader, doBalanceLabels):

        trainFiles = reader.traindata
        triggers = reader.classtrainTriggers
        onsets = reader.classtrainOnsets
        finOnsets = reader.classtrainFinOnsets
        [self.trainData, self.labels] = utilities.dataLoadFromEDF(self, trainFiles, triggers, onsets, finOnsets,
                                                                  self.params)
        if (doBalanceLabels):
            [self.trainData, self.labels] = utilities.balance_labels(self.trainData, self.labels)

        numChannels = self.trainData.shape[1]
        numSamples = self.trainData.shape[0]

        if self.mode == 'IndepChan':
            tmpVec = utilities.get1DFeatures(np.squeeze(self.trainData[0, 0, :]), self.params)
            sumFvecLen = len(tmpVec)
            print(sumFvecLen, self.trainData.shape)
            print(self.trainData[0, 0, :])
        if self.mode == 'CustomFreqs':
            sumFvecLen = len(self.params.channelSelect)
            print(sumFvecLen, '\n', self.params.channelSelect)
        if self.mode == 'Classic':
            sumFvecLen = 0
            for i in range(numChannels):
                if not (self.params.fDiaps is None):
                    self.params.lowFreq = self.params.fDiaps[i][0]
                    self.params.highFreq = self.params.fDiaps[i][1]
                tmpVec = utilities.get1DFeatures(np.squeeze(self.trainData[0, i, :]), self.params)
                sumFvecLen = sumFvecLen + len(tmpVec)
                #print(sumFvecLen, '\n', self.trainData[0, i, :], '\n', tmpVec)

        fVecs = np.zeros((numSamples, numChannels, sumFvecLen))


        #Fastovets
        #indepchan     [nchan = 75 x nvecs = 335 x vecLen = 49] 139
        #classic        [nchan = 75 x nvecs = 335 x vecLen = 49] 2293
        #custom         [nchan = 75 x nvecs = 335 x vecLen = 29]
        #tmpVec.shape = 16


        for k in range(numSamples):  #cust classic = 1
            curFvec = np.zeros((0))
                # print("Fvec: " + str(k) + " of " + str(numSamples))
            for i in range(numChannels):
                if not (self.params.fDiaps is None):
                    self.params.lowFreq = self.params.fDiaps[i][0]
                    self.params.highFreq = self.params.fDiaps[i][1]
                tmpVec = utilities.get1DFeatures(np.squeeze(self.trainData[k, i, :]), self.params)
                if self.mode == 'IndepChan':
                    fVecs[k, i, :] = tmpVec  #"""КАК НАСЧЕТ ТОГО, ЧТОБЫ СОБЛЮДАТЬ РАЗМЕРНОСТЬ FVECS?""" [k, i, :]
                if self.mode == 'CustomFreqs':
                    specVal = np.mean(tmpVec)  # !!!!!!! was not meaned while sumfveclen was counted!
                    fVecs[k, i, 0] = specVal #[k, i]
                if self.mode == 'Classic':
                    curFvec = np.concatenate((curFvec, tmpVec), axis=0)
            if self.mode == 'Classic':
                fVecs[k, i, :] = curFvec #[k, :]

        # Shuffle!
        inds = np.random.permutation(fVecs.shape[0])
        fVecs = fVecs[inds, :, :]
        labels = self.labels[inds]

        #print(self.trainResult.mean, self.trainResult.std)

        # Norm!
        if self.mode == 'Classic' or self.mode == 'CustomFreqs':
            self.trainResult.mean = np.mean(fVecs, 0)
            self.trainResult.std = np.std(fVecs, 0)
            for i in range(fVecs.shape[0]):
                fVecs[i, :, :] = (fVecs[i, :, :] - self.trainResult.mean) / self.trainResult.std
            fTransformed = fVecs
            #PCA
            if self.params.usePCA: #cycle! as in INDEPCHAN
                for i in range(fVecs.shape[1]):
                    tmp = np.zeros((fVecs.shape[0], fVecs.shape(2)))
                    tmp = fVecs(:, i, :)
                    pcaTransform = PCA(self.params.numPC)
                    pcaTransform.fit(tmp)
                    self.trainResult.pcaOp = pcaTransform
                fTransformed = pcaTransform.transform(tmp)


        if self.mode == 'IndepChan':
            for i in range(fVecs.shape[1]):
                chanVecs = np.squeeze(fVecs[:, i, :])
                self.trainResult.mean.append(np.mean(chanVecs, 0))
                self.trainResult.std.append(np.std(chanVecs, 0))
                for k in range(chanVecs.shape[0]):
                    chanVecs[k, :] = (chanVecs[k, :] - self.trainResult.mean[i]) / self.trainResult.std[i]
                fVecs[:, i, :] = chanVecs
            fTransformed = np.zeros((fVecs.shape[0], fVecs.shape[1] * fVecs.shape[2]))
            #PCA
            if (self.params.usePCA):
                fTransformed = np.zeros((fVecs.shape[0], fVecs.shape[1] * self.params.numPC))
            for i in range(numChannels):
                chanVecs = np.squeeze(fVecs[:, i, :])
                if (self.params.usePCA):
                    curPcaTransform = PCA(self.params.numPC)
                    curPcaTransform.fit(chanVecs)
                    self.trainResult.pcaOp.append(curPcaTransform)
                    fTransformed[:, i * self.params.numPC:(i + 1) * self.params.numPC] = curPcaTransform.transform(
                        chanVecs)
                else:
                    fTransformed[:, i * fVecs.shape[2]:(i + 1) * fVecs.shape[2]] = chanVecs

        #LDA
        if not (self.params.finalClassifier is None):
            [Op, fTransformed] = utilities.trainClassifier(self.params.finalClassifier, fTransformed, labels)
            self.trainResult.finalOp = Op
        if not (self.params.distanceFun is None):
            self.trainResult.trainTransformedVecs = fTransformed
        self.trainResult.trainLabels = labels


    def validate(self, reader):
        validateFiles = reader.testdata
        triggers = reader.classtestTriggers
        onsets = reader.classtestOnsets
        finOnsets = reader.classtestFinOnsets
        [validateData, validateLabels] = utilities.dataLoadFromEDF(self, validateFiles, triggers, onsets, finOnsets,
                                                               self.params)
        numChannels = validateData.shape[1]
        numSamples = validateData.shape[0]

        if self.mode == 'IndepChan':
            tmpVec = utilities.get1DFeatures(np.squeeze(validateData[0, 0, :]), self.params)
            sumFvecLen = len(tmpVec)
        if self.mode == 'CustomFreqs':
            sumFvecLen = len(self.params.channelSelect)
        if self.mode == 'Classic':
            sumFvecLen = 0
            for i in range(numChannels):
                if not (self.params.fDiaps is None):
                    self.params.lowFreq = self.params.fDiaps[i][0]
                    self.params.highFreq = self.params.fDiaps[i][1]
                tmpVec = utilities.get1DFeatures(np.squeeze(validateData[0, i, :]), self.params)
                sumFvecLen = sumFvecLen + len(tmpVec)

        fVecs = np.zeros((numSamples, numChannels, sumFvecLen))

        for k in range(numSamples):  #cust classic = 1
            curFvec = np.zeros((0))
                # print("Fvec: " + str(k) + " of " + str(numSamples))
            for i in range(numChannels):
                if not (self.params.fDiaps is None):
                    self.params.lowFreq = self.params.fDiaps[i][0]
                    self.params.highFreq = self.params.fDiaps[i][1]
                tmpVec = utilities.get1DFeatures(np.squeeze(self.trainData[k, i, :]), self.params)
                if self.mode == 'IndepChan':
                    fVecs[k, i, :] = tmpVec  #"""КАК НАСЧЕТ ТОГО, ЧТОБЫ СОБЛЮДАТЬ РАЗМЕРНОСТЬ FVECS?""" [k, i, :]
                if self.mode == 'CustomFreqs':
                    specVal = np.mean(tmpVec)  # !!!!!!! was not meaned while sumfveclen was counted!
                    fVecs[k, i, 0] = specVal #[k, i]
                if self.mode == 'Classic':
                    curFvec = np.concatenate((curFvec, tmpVec), axis=0)
            if self.mode == 'Classic':
                fVecs[k, i, :] = curFvec #[k, :]

        # Norm!
        if self.mode == 'Classic' or self.mode == 'CustomFreqs':
            self.trainResult.mean = np.mean(fVecs, 0)
            self.trainResult.std = np.std(fVecs, 0)
            for i in range(fVecs.shape[0]):
                fVecs[i, :, :] = (fVecs[i, :, :] - self.trainResult.mean) / self.trainResult.std
            fTransformed = fVecs
            # PCA
            if self.params.usePCA:
                fTransformed = self.trainResult.pcaOp.transform(fVecs)
        #Norm!
        if self.mode == 'IndepChan':
            fTransformed = np.zeros((fVecs.shape[0], fVecs.shape[1] * fVecs.shape[2]))
            # PCA
            if (self.params.usePCA):
                fTransformed = np.zeros((fVecs.shape[0], fVecs.shape[1] * self.params.numPC))
            for i in range(numChannels):
                chanVecs = np.squeeze(fVecs[:, i, :])
                for k in range(chanVecs.shape[0]):
                    chanVecs[k, :] = (chanVecs[k, :] - self.trainResult.mean[i]) / self.trainResult.std[i]
                if (self.params.usePCA):
                    fTransformed[:, i * self.params.numPC:(i + 1) * self.params.numPC] = self.trainResult.pcaOp[
                        i].transform(chanVecs)
                else:
                    fTransformed[:, i * fVecs.shape[2]:(i + 1) * fVecs.shape[2]] = chanVecs


        if (not self.params.finalClassifier is None):
            fTransformed = utilities.applyClassifier(self.params.finalClassifier, self.trainResult.finalOp,
                                                     fTransformed)
        result = np.zeros((fTransformed.shape[0]))

        for i, fvec in enumerate(fTransformed):
            if not (self.params.distanceFun is None):
                result[i] = self.params.distanceFun(self.trainResult.trainTransformedVecs, self.trainResult.trainLabels,
                                                    fvec, self.params)
            else:
                result[i] = fTransformed[i]
        classRates, confMat = utilities.calcStats(validateLabels, result)
        # print('Class Rates:\n', classRates)
        # print('Confusion Matrix: \n', confMat)
        return classRates, confMat
