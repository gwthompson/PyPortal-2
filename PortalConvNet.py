from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from PortalParams import Params, TrainResult
import utilities
import numpy as np
import NeuralModels as NM

class Portal_ConvNet():
    def __init__(self, params, neuralFun, numEpochs, batchSize, imgSize):
        self.params = params
        self.params.imgSize = imgSize
        self.trainResult = TrainResult()
        self.trainData = []
        self.labels= np.zeros((0))
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.neuralFun = neuralFun

    def train(self,reader, doBalanceLabels):
        trainFiles = reader.traindata
        triggers = reader.classtrainTriggers
        onsets = reader.classtrainOnsets
        finOnsets = reader.classtrainFinOnsets
        [self.trainData, self.labels] = utilities.dataLoadFromEDF(self, trainFiles, triggers, onsets, finOnsets, self.params)
        self.numClasses = len(set(self.labels))
        if (doBalanceLabels):
            [self.trainData, self.labels] = utilities.balance_labels(self.trainData, self.labels)
        numSamples = self.trainData.shape[0]
        tmpVec = utilities.get2DFeatures(np.squeeze(self.trainData[0, :, :]), self.params)
        if (np.ndim(tmpVec)==3):
            fVecs = np.zeros((numSamples,tmpVec.shape[0],tmpVec.shape[1], tmpVec.shape[2]))
        if (np.ndim(tmpVec)==2):
            fVecs = np.zeros((numSamples,tmpVec.shape[0],tmpVec.shape[1]))
        for k in range(numSamples):
            #print('Train Sample ', k, ' of ', numSamples)
            curFvec = utilities.get2DFeatures(np.squeeze(self.trainData[k, :, :]), self.params)
            if (np.ndim(tmpVec) == 3):
                fVecs[k, :, :, :] = curFvec
            if (np.ndim(tmpVec) == 2):
                fVecs[k, :, :,] = curFvec
        #Shuffle!
        inds = np.random.permutation(fVecs.shape[0])
        if (np.ndim(tmpVec) == 3):
            fVecs = fVecs[inds,:,:,:]
        if (np.ndim(tmpVec) == 2):
            fVecs = fVecs[inds,:,:]
        labels = self.labels[inds]
        # ConvNet!
        model = self.neuralFun(self.numClasses)
        model.trainModel(fVecs,labels,self.batchSize,self.numEpochs)
        self.trainResult.model = model

    def validate(self,reader):
        validateFiles = reader.testdata
        triggers = reader.classtestTriggers
        onsets = reader.classtestOnsets
        finOnsets = reader.classtestFinOnsets
        [validateData, validateLabels] = utilities.dataLoadFromEDF(self, validateFiles, triggers, onsets,finOnsets, self.params)
        numChannels = validateData.shape[1]
        numSamples = validateData.shape[0]
        tmpVec = utilities.get2DFeatures(np.squeeze(self.trainData[0, :, :]), self.params)
        if (np.ndim(tmpVec)==3):
            fVecs = np.zeros((numSamples,tmpVec.shape[0],tmpVec.shape[1], tmpVec.shape[2]))
        if (np.ndim(tmpVec)==2):
            fVecs = np.zeros((numSamples,tmpVec.shape[0],tmpVec.shape[1]))
        result = np.zeros((numSamples))
        for k in range(numSamples):
            #print('Validate Sample ', k, ' of ', numSamples)
            curFvec = utilities.get2DFeatures(np.squeeze(validateData[k, :, :]), self.params)
            result[k] = self.trainResult.model.testModel(curFvec)
        classRates, confMat = utilities.calcStats(validateLabels,result)
        #print('Class Rates:\n', classRates)
        #print('Confusion Matrix: \n', confMat)
        return classRates, confMat