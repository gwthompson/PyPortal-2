from PortalClassicNew import *

from PortalClassic import *
from PortalIndepChan import *
from PortalCustomFreqs import *
from PortalCSP import *
from PortalConvNet import *
import featureFuncs
from os import listdir
from os.path import isfile, join
import xlsxwriter
import adaptivePreprocessing
import NeuralModels
import pickle
from dataReader import edfFileReader
from PortalResExport import ExcelExporter


if __name__ == '__main__':

    #Let's set the algorithm parameters!

    #Sampling Frequency (Hz)
    Fs=500 #mandatory
    #analysis window size (sec)
    winSize=2 #mandatory
    #analysis step size (sec)
    step=1 #mandatory

    #do we use average referencing?
    doAverageReferencing = True #Default: False
    #Resample the data to other Fs
    targetFs = None #default: do not resample
    #List of ints with channel (or ICA component) numbers that we use
    channelSelect = None #Default: all
    #If you calculate spectrum, you may specify frequencies to cut from each of the channel: List[List[minFreq(float), maxFreq(float)]]
    #This can be calculated by adaptivePreprocessing.autoSelectComps
    fDiaps = None #Default: all spectrums will be cut from 1 to 35 Hz
    #do we use PCA
    usePCA = True
    #number of principal Components if used
    numPc = 20 #Default: 5
    #Using Hanning window to smooth spectrum
    doHanning = False #default:False
    #do we use ICA
    useICA = True #default: False
    #Specify ICA fileneme (.mat with 'T' variable inside). If there is no such file, Portal will calculate ICA and generate one
    #if useICA = False, this parameter is inactive
    icaFile = 'T.mat' #default: 'ICA_tmp.mat'
    #The classifier to use at the end. Specify this with the string ('LDA', 'RF', 'MLP' etc)
    finalClassifier = 'MLP' #Default: None, no classifier is used, features directly go to distanceFunc
    #Specify link to feature function that will be used to extract your data features.
    FeatureFunction = featureFuncs.get_spectrum #default: raw EEG will be preserved
    # Specify link to distance function that will be used to calculate distance between features during validation
    DistanceFunction = None #default: no function (in case your classifier outputs labels directly)

    #if we use autoCompsSelect: threshold for component discriminative power (0.5-0.95)
    #This can iteratively decrease by 0.05 if not enough powerful components are found
    compThresh = 0.65 #mandatory
    #How many components at least should be found?
    minCompsNum = numPc #Default - either 2 or, if usePCA = True, minCompsNum = pcaNUM+1


    Exporter = ExcelExporter()
    subjects=[]
    classRates2 = []

    genpath = '/home/ubuntu/Desktop/EEG/BigExp/DATA_NO_IMAG'
    row=0
    success = 0
    allCount = 0
    subjects = [d for d in os.listdir(genpath) if os.path.isdir(os.path.join(genpath, d))]

    for subj in subjects:
        pc = edfFileReader(genpath, subj = subj, chansToDrop=['Diff 2', 'Diff 3', 'Diff 4', 'EDF Annotations'],
                        trainDir='Session1', testDir='Session2', icaFilename = icaFile, commonLabelsPath='labels.txt')
        pc.readData(startOffset=0.5, mixData=True)
        PortalParams = Params(Fs=pc.Fs, winSize=winSize, step=step, usePCA=usePCA, numPc=numPc,
                              doAverageReferencing=doAverageReferencing, channelSelect=channelSelect, fDiaps=fDiaps,
                              useICA=useICA, icaFile=pc.icaDir, finalClassifier=finalClassifier, doHanning=doHanning,
                              distanceFun=DistanceFunction, featureFun=FeatureFunction, sensorsPosFile=pc.senLocDir)
        print('Analysing Subject: ' + subj)
        row=row+1

        # PortalWorker = Portal_ConvNet(PortalParams, neuralFun=NeuralModels.Conv1DModel, numEpochs=100, batchSize=8, imgSize=64)
        PortalWorker = Portal_Classic(PortalParams)
        #PortalWorker = Portal_Classic_New(PortalParams, customFreq=False)
        # print('Selecting Components...')
        [comps, diaps] = adaptivePreprocessing.autoSelectComps(PortalWorker.params, pc, compThresh=compThresh, minCompsNum=minCompsNum)

        PortalWorker.params.setWinSizeAndStep(winSize = 3, step = 0.5)
        PortalWorker.params.channelSelect = comps
        PortalWorker.params.fDiaps = diaps

        Exporter.addNewSubject(subj)
        n=0
        for i in range(0,5):
            for j in range(1,5):
                if i<j:
                    print('Training On Subject: ' + subj + ' ' + str(i + 1) + ' vs ' + str(j + 1))
                    nameForTable = str(i) + '-' + str(j)
                    pc.setClasses(classesToProcess=[i, j])
                    PortalWorker.train(reader=pc,doBalanceLabels=True)
                    classRates_, confMat_ = PortalWorker.validate(reader=pc, )
                    val1 = int(np.round(100*classRates_[0]))
                    val2 = int(np.round(100*classRates_[1]))
                    Exporter.addPairDataField(nameForTable,val1,val2)
                    if (classRates_[0]>0.5) and (classRates_[1]>0.5):
                        classRates2.append(classRates_[0])
                        classRates2.append(classRates_[1])
                        success=success+1
                    n=n+1
                    allCount=allCount+1
    row=row+2
    average = 100*np.mean(np.array(classRates2))
    Exporter.addFinisher('Valid: ', (success/float(allCount)))
    Exporter.addFinisher('Acc: ',average)
    Exporter.Finish()