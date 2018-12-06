import os
from mne.io import read_raw_edf, RawArray
import numpy as np
from mne import create_info

class edfFileReader:
    def __init__(self, basedir, subj, chansToDrop=None, trainDir=None, testDir=None, icaFilename = None, commonLabelsPath=None,
                 commonOnsetsPath = None, commonFinOnsetsPath=None):
        self.basedir = basedir
        pardir = os.path.abspath(os.path.join(self.basedir, os.pardir))
        self.traindir = self.basedir + '//' + subj + '//'
        self.subjDir = self.basedir + '//' + subj + '//'
        self.testdir = self.basedir + '//' + subj + '//'
        if not (trainDir is None):
            self.traindir = self.basedir + '//' + subj + '//' + trainDir + '//'
        if not (testDir is None):
            self.testdir = self.basedir + '//' + subj + '//' + testDir + '//'
        self.icaDir = None
        if not(icaFilename is None):
            self.icaDir = self.basedir + '//' + subj + '//' + icaFilename
        self.senLocDir = basedir + '//sensorsPosition.txt'
        self.trainfiles=[]
        self.testfiles=[]
        self.trainTriggerFiles = []
        self.testTriggerFiles = []
        self.trainOnsetFiles = []
        self.testOnsetFiles = []
        self.trainFinOnsetFiles = []
        self.testFinOnsetFiles = []

        for file in os.listdir(self.traindir):
            if file.endswith(".edf"):
                self.trainfiles.append(self.traindir + file)
                if not (commonLabelsPath is None):
                    self.trainTriggerFiles.append(self.basedir + '//' + subj + '//' + commonLabelsPath)
                else:
                    self.trainTriggerFiles.append(self.traindir + file[:-4]+'_labels.txt')
                if not (commonOnsetsPath is None):
                    self.trainOnsetFiles.append(self.basedir + '//' + subj + '//' + commonOnsetsPath)
                else:
                    self.trainOnsetFiles.append(self.traindir + file[:-4] + '_onsets_start.txt')
                if not (commonFinOnsetsPath is None):
                    self.trainFinOnsetFiles.append(self.basedir + '//' + subj + '//' + commonFinOnsetsPath)
                else:
                    self.trainFinOnsetFiles.append(self.traindir + file[:-4] + '_onsets_end.txt')
        self.trainfiles = [read_raw_edf(f, preload=True, stim_channel=None) for f in
                           self.trainfiles]
        if not (chansToDrop is None):
            for raw in self.trainfiles:
                try:
                    raw.drop_channels(chansToDrop)
                except:
                    ValueError('Incorrectly specified names of channels to drop!')

        if not (testDir is None):
            for file in os.listdir(self.testdir):
                if file.endswith(".edf"):
                    self.testfiles.append(self.testdir + file)
                    if not (commonLabelsPath is None):
                        self.testTriggerFiles.append(self.basedir + '//' + subj + '//' + commonLabelsPath)
                    else:
                        self.testTriggerFiles.append(self.testdir + file[:-4] + '_labels.txt')
                    if not (commonOnsetsPath is None):
                        self.testOnsetFiles.append(self.basedir + '//' + subj + '//' + commonOnsetsPath)
                    else:
                        self.testOnsetFiles.append(self.testdir + file[:-4] + '_onsets_start.txt')
                    if not (commonFinOnsetsPath is None):
                        self.testFinOnsetFiles.append(self.basedir + '//' + subj + '//' + commonFinOnsetsPath)
                    else:
                        self.testFinOnsetFiles.append(self.testdir + file[:-4] + '_onsets_end.txt')

            self.testfiles = [read_raw_edf(f, preload=True, stim_channel=None) for f in
                              self.testfiles]
            for raw in self.testfiles:
                try:
                    raw.drop_channels(chansToDrop)
                except:
                    ValueError('Incorrectly specified names of channels to drop!')

    def readData(self, startOffset=0, validationPart = 0.5, mixData = True):
        self.Fs = self.trainfiles[0].info['sfreq']
        self.trainTriggers=[]
        self.testTriggers=[]
        self.trainOnsets = []
        self.testOnsets=[]
        self.trainFinOnsets=[]
        self.testFinOnsets=[]
        self.traindata = []
        self.testdata=[]
        for i in range(len(self.trainfiles)):
            lab_file = open(self.trainTriggerFiles[i], "r")
            ftriggers = list(map(int, lab_file.readlines()))
            lab_file.close()
            onset_file = open(self.trainOnsetFiles[i], "r")
            onsets = list(map(float, onset_file.readlines()))
            onset_end_file = open(self.trainFinOnsetFiles[i], "r")
            fin_onsets = list(map(float, onset_end_file.readlines()))
            onset_file.close()
            onset_end_file.close()
            lab_file.close()
            self.trainTriggers.append(ftriggers)
            self.trainOnsets.append(onsets)
            self.trainFinOnsets.append(fin_onsets)
        for i in range(len(self.testfiles)):
            lab_file = open(self.testTriggerFiles[i], "r")
            ftriggers = list(map(int, lab_file.readlines()))
            lab_file.close()
            onset_file = open(self.testOnsetFiles[i], "r")
            onsets = list(map(float, onset_file.readlines()))
            onset_end_file = open(self.testFinOnsetFiles[i], "r")
            fin_onsets = list(map(float, onset_end_file.readlines()))
            onset_file.close()
            onset_end_file.close()
            lab_file.close()
            self.testTriggers.append(ftriggers)
            self.testOnsets.append(onsets)
            self.testFinOnsets.append(fin_onsets)
        if (mixData==True):
            self.trainfiles+=self.testfiles
            self.trainTriggers+=self.testTriggers
            self.testOnsets+=self.testOnsets
            self.trainFinOnsets+=self.testFinOnsets
            self.testfiles=[]

        self.trainTestSplit(startOffset=startOffset, validationPart=validationPart)
        return

    def trainTestSplit(self, validationPart=0.5, startOffset = 0.0):
        allOnsets=[]
        allFinOnsets=[]
        allTriggers=[]
        nChans = self.trainfiles[0].info['nchan']
        allData = np.zeros((nChans, 0))
        allFiles = self.trainfiles+self.testfiles
        self.trainFinOnsets+=self.testFinOnsets
        self.trainTriggers+=self.testTriggers
        self.trainOnsets+=self.testOnsets
        for nf in range(len(allFiles)):
            fileOnsets = self.trainOnsets[nf]
            fileFinishOnsets = self.trainFinOnsets[nf]
            fileTriggers = self.trainTriggers[nf]
            curRawTrainData = allFiles[nf].get_data()
            for nt in range(len(fileTriggers)):
                allOnsets.append(int(allData.shape[1]))
                allTriggers.append(fileTriggers[nt])
                startSample = int(np.round(fileOnsets[nt]*self.Fs)+np.round(startOffset*self.Fs))
                endSample = int(np.round(fileFinishOnsets[nt]*self.Fs))
                curTrainData = curRawTrainData[:, startSample:endSample]
                allData = np.concatenate((allData, curTrainData), axis=1)
                allFinOnsets.append(int(allData.shape[1]-1))
            if (nf==len(self.trainfiles)-1) and (len(self.testfiles)>0):
                raw_array = RawArray(allData, create_info(self.trainfiles[0].info['ch_names'], sfreq=self.Fs),
                                     verbose='ERROR')
                self.traindata = raw_array
                trainTriggers = allTriggers
                trainOnsets = allOnsets
                trainFinOnsets = allFinOnsets
                allOnsets = []
                allFinOnsets = []
                allTriggers = []
                allData = np.zeros((nChans, 0))
        if len(self.testfiles):
            #If we have specified train and test data, then finish here
            raw_array = RawArray(allData, create_info(self.trainfiles[0].info['ch_names'], sfreq=self.Fs),
                                 verbose='ERROR')
            self.trainTriggers = trainTriggers
            self.trainOnsets = trainOnsets
            self.trainFinOnsets = trainFinOnsets
            self.testdata = raw_array
            self.testTriggers = allTriggers
            self.testOnsets = allOnsets
            self.testFinOnsets = allFinOnsets
            return
        #else = we shuffle!
        np.random.seed(83)
        allOnsets = np.array(allOnsets)
        allFinOnsets = np.array(allFinOnsets)
        allTriggers = np.array(allTriggers)
        numClasses = len(np.unique(allTriggers))
        trainTriggers=np.zeros((0,))
        testTriggers=np.zeros((0,))
        trainOnsets=np.zeros((0,))
        testOnsets=np.zeros((0,))
        trainFinOnsets=np.zeros((0,))
        testFinOnsets=np.zeros((0,))
        for i in range(numClasses):
            mask = (allTriggers == i)
            classTriggers = allTriggers[mask]
            classOnsets = allOnsets[mask]
            classFinOnsets = allFinOnsets[mask]
            N = classTriggers.shape[0]
            inds = np.random.permutation(N)
            hSize = int(N * validationPart)
            trainTriggers = np.concatenate((trainTriggers, classTriggers[inds[0:hSize]]))
            testTriggers = np.concatenate((testTriggers, classTriggers[inds[hSize:]]))
            trainOnsets = np.concatenate((trainOnsets, classOnsets[inds[0:hSize]]))
            testOnsets = np.concatenate((testOnsets, classOnsets[inds[hSize:]]))
            trainFinOnsets = np.concatenate((trainFinOnsets, classFinOnsets[inds[0:hSize]]))
            testFinOnsets = np.concatenate((testFinOnsets, classFinOnsets[inds[hSize:]]))

        raw_array = RawArray(allData, create_info(self.trainfiles[0].info['ch_names'], sfreq=self.Fs), verbose='ERROR')
        self.traindata = raw_array
        self.testdata = raw_array
        self.trainTriggers = trainTriggers
        self.testTriggers = testTriggers
        self.trainOnsets = list(trainOnsets.astype(int))
        self.testOnsets = list(testOnsets.astype(int))
        self.trainFinOnsets = list(trainFinOnsets.astype(int))
        self.testFinOnsets = list(testFinOnsets.astype(int))

        return

    def setClasses(self, classesToProcess):
        if not (classesToProcess is None):
            classesToProcess=np.array(classesToProcess)
            classesToProcess = np.sort(classesToProcess)
            maskTrain = np.zeros((len(self.trainTriggers)),dtype=bool)
            maskTest = np.zeros((len(self.testTriggers)), dtype=bool)
            for nClass in classesToProcess:
                maskTrain = maskTrain | (np.array(self.trainTriggers) == nClass)
                maskTest = maskTest | (np.array(self.testTriggers) == nClass)

            self.classtrainOnsets = list(np.array(self.trainOnsets)[maskTrain])
            self.classtrainFinOnsets = list(np.array(self.trainFinOnsets)[maskTrain])
            self.classtestOnsets = list(np.array(self.testOnsets)[maskTest])
            self.classtestFinOnsets = list(np.array(self.testFinOnsets)[maskTest])
            self.classtrainTriggers = np.array(self.trainTriggers)[maskTrain]
            self.classtestTriggers = np.array(self.testTriggers)[maskTest]

            for i, nClass in enumerate(classesToProcess):
                self.classtrainTriggers[self.classtrainTriggers == nClass] = i
                self.classtestTriggers[self.classtestTriggers == nClass] = i
            self.classtrainTriggers = list(self.classtrainTriggers)
            self.classtestTriggers = list(self.classtestTriggers)