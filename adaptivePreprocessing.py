import utilities
import numpy as np
import os
import scipy.io
import math
from scipy.fftpack import fftfreq
from featureFuncs import get_spectrum

def calcAndSaveICA(ClassifierParams, reader, icaMatFileName=None):
    trainFiles = reader.traindata
    triggers = reader.trainTriggers
    onsets = reader.trainOnsets
    finOnsets = reader.trainFinOnsets
    trainData, labels, trainonsets, trainfinOnsets = utilities.continuousFromEDF(trainFiles, triggers, onsets,
                                                                                 finOnsets, ClassifierParams)
    print('Calculating ICA...')
    if (icaMatFileName is None):
        icaMatFileName = reader.subjDir + 'T.mat'
    icaTransf = utilities.trainIca(trainData, maxIter=100)
    icaMat = icaTransf.components_
    td = dict()
    td['T'] = icaMat
    scipy.io.savemat(icaMatFileName, td)
    print('Saved ICA!')

def autoSelectComps(ClassifierParams, reader, compThresh, minCompsNum = None):
    trainFiles = reader.traindata
    triggers = reader.trainTriggers
    onsets = reader.trainOnsets
    finOnsets = reader.trainFinOnsets

    trainData, labels, trainonsets, trainfinOnsets = utilities.continuousFromEDF(trainFiles, triggers, onsets,
                                                                                 finOnsets, ClassifierParams)
    useIca = ClassifierParams.useICA
    if (minCompsNum is None):
        minCompsNum = 2
    # if ClassifierParams.usePCA:
    #     if not (ClassifierParams.numPC is None):
    #         if (minCompsNum<ClassifierParams.numPC):
    #             minCompsNum = ClassifierParams.numPC + 1
    icaMatFileName = ClassifierParams.icaFile
    if useIca:
        if (icaMatFileName is None) or not (os.path.isfile(icaMatFileName)):
            calcAndSaveICA(ClassifierParams, reader, icaMatFileName)
        icaMat = scipy.io.loadmat(icaMatFileName)['T']
        dataToProcess_ = np.matmul(icaMat, trainData)
        print('Got ICA!')
    else:
        dataToProcess_ = trainData
    [trainData, labels] = utilities.dataLoadFromContinuous(ClassifierParams, dataToProcess_, labels, trainonsets,
                                                                     trainfinOnsets)

    numChannels = trainData.shape[1]
    numSamples = trainData.shape[0]
    tmpVec = utilities.get1DFeatures(np.squeeze(trainData[0, 0, :]), ClassifierParams)
    fvecLen = len(tmpVec)
    fVecs_ = np.zeros((numSamples, numChannels, fvecLen))
    for k in range(numSamples):
        # print("Fvec: " + str(k) + " of " + str(numSamples))
        for i in range(numChannels):
            fVecs_[k, i, :] = utilities.get1DFeatures(np.squeeze(trainData[k, i, :]),
                                                      ClassifierParams)

    # choosing components
    N = trainData.shape[2]
    NFFT = int(math.pow(2, int(math.ceil(math.log(N, 2)))))
    T = 1.0 / ClassifierParams.Fs
    freqs = fftfreq(NFFT, T)
    freqs = freqs[1:NFFT // 2]
    idx = np.where((freqs > ClassifierParams.lowFreq) * (freqs < ClassifierParams.highFreq))
    freqs = freqs[idx]
    print('Calcuating individual components and frequencies...')
    comps, diaps = autoSpectralChoose(fVecs_, labels, freqs, minFreq=1.5, maxFreq=18, freqWin=3, t=compThresh, minCompsNum=minCompsNum)
    return comps, diaps

def autoSpectralChoose(trainData, trainLabels, freqs, minFreq, maxFreq, freqWin, t, minCompsNum = 5):
    f2 = freqs[freqs <= maxFreq]
    f2 = f2[f2>=minFreq]
    specWinSize = len(f2[f2<=f2[0]+freqWin])
    numClasses = len(np.unique(trainLabels))
    numComps = trainData.shape[1]
    comps=[]
    diaps=[]
    for nComp in range(numComps):
        #print('Choosing comps: ' + str(nComp) + ' of ' + str(numComps))
        classData=[]
        for nClass in range(numClasses):
            classData.append(trainData[trainLabels==nClass,nComp,:])
        goodFreqs = np.zeros((len(f2)))
        for sw in range (0,len(f2) - specWinSize):
            fdiap = np.arange(sw,sw+specWinSize)
            good = checkValid(classData, fdiap, t)
            if (good):
                goodFreqs[sw:sw + specWinSize] = 1
        if (sum(goodFreqs) > 1):
            curDiaps = getDiaps(goodFreqs,f2)
            for d in curDiaps:
                comps.append(nComp)
                diaps.append(d)
    if (len(comps)<minCompsNum):
        print ('WARNING! Number of comps too low (' + str(len(comps)) + '<' + str(minCompsNum) + '). Lowering threshold by 0.05')
        t=t-0.05
        [comps,diaps] = autoSpectralChoose(trainData, trainLabels, freqs, minFreq, maxFreq, freqWin, t, minCompsNum)
    return comps, diaps

def checkValid(continuousData, fdiap, t):
    good = 0
    meanData = np.zeros((len(continuousData)))
    classData = []
    for i, curClassData in enumerate(continuousData):
        samplesData = np.squeeze(np.mean(curClassData[:, fdiap],1))
        outliers = np.ones((1, samplesData.shape[0]))
        n = 0
        while (np.sum(outliers)) > 0:
            skoAll = np.std(samplesData)
            outliers = (samplesData > np.median(samplesData) + 3.5 * skoAll).astype(int)
            samplesData[outliers>0] = np.median(samplesData)
            n = n + 1
            if (n > 5):
                break
        meanData[i] = np.mean(samplesData)
        classData.append(samplesData)
    mVal = np.mean(meanData)
    minInd = int(np.argmin(meanData))
    maxInd = int(np.argmax(meanData))
    class1Data = classData[minInd]
    class2Data = classData[maxInd]
    m1 = np.mean(class1Data)
    m2 = np.mean(class2Data)
    thr = (m2 + m1) / 2
    lev1 = np.sum(class1Data < thr) / (class1Data.shape[0])
    lev2 = np.sum(class2Data > thr) / (class2Data.shape[0])
    if (lev1 > t) & (lev2 > t):
        good = 1
    return good

def getDiaps(rawDiap, freqs):
    diaps=[]
    prev = 0
    curDiap = [0, 0]
    for i in range(rawDiap.shape[0]):
        if ((rawDiap[i]==1) & (prev==0)):
            curDiap[0] = np.floor(freqs[i])
            prev=1
        elif ((rawDiap[i]==0 or (i==(rawDiap.shape[0]-1))) and (prev==1)):
            curDiap[1] = np.ceil(freqs[i-1])
            diaps.append(curDiap)
            curDiap=[0,0]
            prev=0
    return diaps