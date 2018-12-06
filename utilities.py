from sklearn.metrics import confusion_matrix
import os
from mne.io import concatenate_raws, read_raw_edf
import numpy as np
from sklearn.decomposition import FastICA
import time
import scipy.io
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


def dataLoadFromEDF(Classifier, raw_file,triggers,onsets, finishOnsets, params):
    resampleRate = float(params.rawFs) / float(params.Fs)
    if not (params.rawFs == params.Fs):
        raw_file = raw_file.resample(params.Fs, npad='auto')
    onsets = (np.array(onsets) / resampleRate).astype(int)
    finishOnsets = (np.array(finishOnsets) / resampleRate).astype(int)
    curRawTrainData = raw_file.get_data()
    if (params.doAverageReferencing):
        average = np.mean(curRawTrainData, axis=0)
        for i in range(curRawTrainData.shape[0]):
            curRawTrainData[i, :] = (curRawTrainData[i, :] - average)
    curRawTrainData = curRawTrainData * 1000000
    if params.useICA:
        if (os.path.isfile(params.icaFile)):
            icaMat = scipy.io.loadmat(params.icaFile)['T']
            curRawTrainData = np.matmul(icaMat,curRawTrainData)
        else:
            FileExistsError('No ICA file found, however, ICA requested')
    if not Classifier.params.channelSelect==[]:
        curRawTrainData = curRawTrainData[Classifier.params.channelSelect,:]
    for nt in range(len(triggers)):
        startSample = onsets[nt]
        endSample = finishOnsets[nt]
        curTrainData = curRawTrainData[:,startSample:endSample]
        samples = sampleData(curTrainData, Classifier.params)
        if (nt == 0):
            trainData = samples
            labels = np.zeros((samples.shape[0])) + triggers[nt]
        else:
            trainData = np.concatenate([trainData, samples], 0)
            labels = np.concatenate([labels, np.zeros((samples.shape[0])) + triggers[nt]])
    return trainData, labels

def preprocessDataChunk(input, Classifier):
    curRawTrainData = input
    curRawTrainData = curRawTrainData * 1000000
    if Classifier.params.useICA:
        if (os.path.isfile(Classifier.params.icaFile)):
            icaMat = scipy.io.loadmat(Classifier.params.icaFile)['T']
            curRawTrainData = np.matmul(icaMat, curRawTrainData)
        else:
            FileExistsError('No ICA file found, however, ICA requested')
    if not Classifier.params.channelSelect == []:
        curRawTrainData = curRawTrainData[Classifier.params.channelSelect, :]
    if curRawTrainData.shape[1]<Classifier.params.winSize:
        AssertionError('Data Chunk too small (check winSize)')
    if curRawTrainData.shape[1] > Classifier.params.winSize:
        curRawTrainData = curRawTrainData[:,0:Classifier.params.winSize]
    return curRawTrainData

def continuousFromEDF(raw_data,triggers,onsets, finishOnsets, params):
    resampleRate = float(params.rawFs) / float(params.Fs)
    if not (params.rawFs == params.Fs):
        raw_data = raw_data.resample(params.Fs, npad='auto')
    onsets = (np.array(onsets) / resampleRate).astype(int)
    finishOnsets = (np.array(finishOnsets) / resampleRate).astype(int)
    curRawTrainData = raw_data.get_data()
    if (params.doAverageReferencing):
        average = np.mean(curRawTrainData, axis=0)
        for i in range(curRawTrainData.shape[0]):
            curRawTrainData[i, :] = (curRawTrainData[i, :] - average)
    curRawTrainData = curRawTrainData * 1000000
    trainData = curRawTrainData
    return trainData, triggers, onsets, finishOnsets

def dataLoadFromContinuous(ClassifierParams, data,triggers,onsets, finishOnsets):
    for nt in range(len(triggers)):
        startSample = onsets[nt]
        endSample = finishOnsets[nt]
        curTrainData = data[:,startSample:endSample]
        samples = sampleData(curTrainData, ClassifierParams)
        if (nt == 0):
            trainData = samples
            labels = np.zeros((samples.shape[0])) + triggers[nt]
        else:
            trainData = np.concatenate([trainData, samples], 0)
            labels = np.concatenate([labels, np.zeros((samples.shape[0])) + triggers[nt]])
    return trainData, labels

def sampleData(sig,params):
    numTimestamps = sig.shape[1]
    numChannels = sig.shape[0]
    startSample = 0
    endSample = params.winSize
    numSamples = int(np.floor((numTimestamps - params.winSize - 1)/params.step) + 1)
    if (numSamples<2):
        aaa=1
    dataSamples = np.zeros((numSamples, numChannels, params.winSize))
    for i in range(numSamples):
        curSamples = np.array(sig[:,startSample:endSample])
        startSample = startSample + params.step
        endSample = endSample + params.step
        dataSamples[i,:,:] = curSamples
    return dataSamples

def get1DFeatures(dataSample, params):
    if not (params.featureFun is None):
        if (np.ndim(dataSample)==1):
            arr = np.array(params.featureFun(dataSample, params))
            return arr.ravel()
        if (np.ndim(dataSample) == 2):
            res=[]
            for dsamp in dataSample:
                arr = np.array(params.featureFun(dsamp, params))
                res.append(arr.ravel())
            return np.array(res)
        raise ValueError('Wrong number of sample dimensions!')
    return dataSample

def get2DFeatures(dataSample, params):
    if not (params.featureFun is None):
        if (np.ndim(dataSample)==1):
            raise ValueError('Wrong number of sample dimensions!')
        if (np.ndim(dataSample) == 2):
            arr = np.array(params.featureFun(dataSample, params))
            return np.array(arr)
        raise ValueError('Wrong number of sample dimensions!')
    return dataSample

def eucDist(x1,x2):
    return np.sum(pow((x1-x2),2))

def calcStats(gTruth, output):
    numClasses = len(set(gTruth))
    classRates = np.zeros((numClasses))
    classSums = np.zeros((numClasses))
    for i in range (numClasses):
        for j in range (len(gTruth)):
            if gTruth[j] == i:
                classSums[i] = classSums[i] + 1
                if output[j] == i:
                    classRates[i] = classRates[i] + 1
    classRates = classRates / classSums
    confMat = confusion_matrix(gTruth, output)
    return classRates, confMat

def balance_labels(data, labels):
    numClasses = len(set(labels))
    samplesInClass = np.zeros((numClasses))
    np.random.seed(10)
    for i in range(numClasses):
        samplesInClass[i] = np.sum(labels == i)
    nSamples = np.min(samplesInClass)
    deleted=[]
    for i in range(numClasses):
        sInClass = np.sum(labels == i)
        while(sInClass>nSamples):
            mask = (labels==i)
            inds = np.squeeze(np.argwhere(mask))
            delInd = np.random.randint(0,len(inds)-1)
            while inds[delInd] in deleted:
                delInd = np.random.randint(0, len(inds) - 1)
            deleted.append(inds[delInd])
            sInClass = sInClass - 1
    data = np.delete(data, np.array(deleted), 0)
    labels = np.delete(labels,np.array(deleted))
    return data, labels

def trainIca(data,maxIter = 200):
    data = np.transpose(data)
    print('Calculating ICA for ' + str(maxIter) + ' iterations')
    transformer = FastICA(random_state=0, max_iter=maxIter)
    now = time.time()
    transformer.fit(data)
    then = time.time()
    elapsed = then-now
    return transformer

def trainClassifier(classifierType, fvecs, labels):
    if (classifierType == 'LDA'):
        finalTransform = LinearDiscriminantAnalysis()
        finalTransform.fit(fvecs, labels)
        Op = finalTransform
        fTransformed = finalTransform.transform(fvecs)
    if (classifierType == 'RF'):
        rfTransform = RandomForestClassifier(n_estimators=50)
        rfTransform.fit(fvecs, labels)
        fTransformed = -1
        Op = rfTransform
    if (classifierType == 'LinearSVM'):
        Op = SVC(kernel='linear', gamma='auto')
        Op.fit(fvecs,labels)
        fTransformed = -1
    if (classifierType == 'RBFSVM'):
        Op = SVC(kernel='rbf', gamma='auto')
        Op.fit(fvecs,labels)
        fTransformed = -1
    if (classifierType == 'NaiveBayes'):
        Op = GaussianNB()
        Op.fit(fvecs,labels)
        fTransformed = -1
    if (classifierType=='MLP'):
        hlsize = int(fvecs.shape[1]/2)
        if (hlsize<10):
            hlsize=10
        Op = MLPClassifier(hidden_layer_sizes=(hlsize,))
        Op.fit(fvecs,labels)
        fTransformed=-1
    return Op, fTransformed

def applyClassifier(classifierType, Op, trainvecs):
    if (classifierType == 'LDA'):
        fTransformed = Op.transform(trainvecs)
    if (classifierType == 'RF'):
        fTransformed = Op.predict(trainvecs)
    if ((classifierType == 'LinearSVM') or (classifierType == 'RBFSVM')):
        fTransformed = Op.predict(trainvecs)
    if (classifierType == 'NaiveBayes'):
        fTransformed = Op.predict(trainvecs)
    if (classifierType == 'MLP'):
        fTransformed = Op.predict(trainvecs)
    return fTransformed