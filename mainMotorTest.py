from PortalClassic import *
from PortalIndepChan import *
from PortalCustomFreqs import *
from PortalCSP import *
from PortalConvNet import *
import featureFuncs
from os import listdir
import distanseFuncs
from os.path import isfile, join
import scipy.io
from mne.io import concatenate_raws, read_raw_edf
import xlsxwriter
import adaptivePreprocessing
import NeuralModels
from helpers import trainTestSplit


class mainTest:
    def __init__(self, basedir, subj):
        self.basedir = basedir
        pardir = os.path.abspath(os.path.join(self.basedir, os.pardir))
        self.traindir = self.basedir + '//' + subj + '//'
        self.testdir = self.basedir + '//' + subj + '//'
        self.icaDir = self.basedir + '//' + subj + '//TPython_100.mat'
        self.labFile = pardir + '//labels.txt'
        trainfiles = [self.traindir + 'Session1.edf']
        testfiles =  [self.traindir + 'Session2.edf']
        self.trainfiles = [read_raw_edf(f, preload=True, stim_channel=None) for f in
                           trainfiles]
        self.testfiles = [read_raw_edf(f, preload=True, stim_channel=None) for f in
                          testfiles]
        for raw in self.trainfiles:
            try:
                raw.drop_channels(['Diff 2', 'Diff 3', 'Diff 4', 'EDF Annotations'])
            except:
                raw.drop_channels(['EDF Annotations'])
        for raw in self.testfiles:
            try:
                raw.drop_channels(['Diff 2', 'Diff 3', 'Diff 4', 'EDF Annotations'])
            except:
                raw.drop_channels(['EDF Annotations'])

    def setParams(self, class1=None, class2=None):

        lab_file = open(self.labFile, "r")
        ftriggers = list(map(int, lab_file.readlines()))
        ftriggers = list([ftriggers,ftriggers])
        lab_file.close()
        onset_file_1 = open(self.traindir + "onsets_start_1.txt", "r")
        onset_end_file_1 = open(self.traindir + "onsets_end_1.txt", "r")
        onset_file_2 = open(self.testdir + "onsets_start_2.txt", "r")
        onset_end_file_2 = open(self.testdir + "onsets_end_2.txt", "r")
        fonsets_1 = list(map(float, onset_file_1.readlines()))
        fonsets_2 = list(map(float, onset_file_2.readlines()))
        fonsets = list([fonsets_1, fonsets_2])
        fin_onsets_1 = list(map(float, onset_end_file_1.readlines()))
        fin_onsets_2 = list(map(float, onset_end_file_2.readlines()))
        fonsets_end = list([fin_onsets_1, fin_onsets_2])
        onset_file_1.close()
        onset_file_2.close()
        onset_end_file_1.close()
        onset_end_file_2.close()
        Fs = self.trainfiles[0].info['sfreq']
        startOffset = 0.5

        [raw_array, trainTriggers, testTriggers, trainOnsets, testOnsets, trainFinOnsets, testFinOnsets]=\
            trainTestSplit(self.trainfiles, fonsets, fonsets_end, ftriggers, Fs, validationPart=0.5, startOffset=startOffset)

        self.cutfiles = list([raw_array])

        finOnsetsAllClasses = np.concatenate((trainFinOnsets, testFinOnsets))
        fonsetsAllClasses = np.concatenate((trainOnsets, testOnsets))
        ftriggersAllClasses = np.concatenate((trainTriggers, testTriggers))
        if not (class1 is None):
            maskTrain = (trainTriggers == class1) | (trainTriggers == class2)
            trainOnsets = trainOnsets[maskTrain]
            trainFinOnsets = trainFinOnsets[maskTrain]
            maskTest = (testTriggers == class1) | (testTriggers == class2)
            testOnsets = testOnsets[maskTest]
            testFinOnsets = testFinOnsets[maskTest]

            trainTriggers = trainTriggers[maskTrain]
            trainTriggers[trainTriggers == class1] = 0
            trainTriggers[trainTriggers == class2] = 1
            testTriggers = testTriggers[maskTest]
            testTriggers[testTriggers == class1] = 0
            testTriggers[testTriggers == class2] = 1

        self.trainTriggers = np.expand_dims(trainTriggers,0)
        self.testTriggers = np.expand_dims(testTriggers,0)
        self.trainTriggersAllClasses = np.expand_dims(ftriggersAllClasses,0)
        self.trainOnsetsAllClasses = np.expand_dims(fonsetsAllClasses,0)
        self.trainFinOnsetsAllClasses = np.expand_dims(finOnsetsAllClasses,0)
        self.trainOnsets = np.expand_dims(trainOnsets,0)
        self.testOnsets = np.expand_dims(testOnsets, 0)
        self.finTrainOnsets = np.expand_dims(trainFinOnsets, 0)
        self.finTestOnsets = np.expand_dims(testFinOnsets, 0)
        return 83

    def setParamsFor2Files(self, class1, class2):

        lab_file = open(self.labFile, "r")
        ftriggers = list(map(int, lab_file.readlines()))
        ftriggers = np.array(ftriggers)
        lab_file.close()
        train_onset_file = open(self.traindir + "onsets_start_1.txt", "r")
        train_onset_end_file = open(self.traindir + "onsets_end_1.txt", "r")
        test_onset_file = open(self.testdir + "onsets_start_2.txt", "r")
        test_onset_end_file = open(self.testdir + "onsets_end_2.txt", "r")

        train_fonsets = list(map(float, train_onset_file.readlines()))
        train_fonsets = np.array(train_fonsets)
        train_onset_file.close()
        train_fonsets_end = list(map(float, train_onset_end_file.readlines()))
        train_fonsets_end = np.array(train_fonsets_end)
        train_onset_end_file.close()

        test_fonsets = list(map(float, test_onset_file.readlines()))
        test_fonsets = np.array(test_fonsets)
        test_onset_file.close()
        test_fonsets_end = list(map(float, test_onset_end_file.readlines()))
        test_fonsets_end = np.array(test_fonsets_end)
        test_onset_end_file.close()

        Fs = self.trainfiles[0].info['sfreq']
        startOffset = np.round(0.5 * Fs)
        train_fonsets = np.round(train_fonsets * Fs).astype(int)
        train_finOnsets = np.round(train_fonsets_end * Fs).astype(int)
        train_fonsets = train_fonsets + startOffset
        test_fonsets = np.round(test_fonsets * Fs).astype(int)
        test_finOnsets = np.round(test_fonsets_end * Fs).astype(int)
        test_fonsets = test_fonsets + startOffset
        ftriggersAllClasses = ftriggers
        train_finOnsetsAllClasses = train_finOnsets
        train_fonsetsAllClasses = train_fonsets
        mask = (ftriggers == class1) | (ftriggers == class2)
        train_fonsets = train_fonsets[mask]
        train_finOnsets = train_finOnsets[mask]
        test_fonsets = test_fonsets[mask]
        test_finOnsets = test_finOnsets[mask]
        # imag_fonsets = imag_fonsets[maskImag]
        # imag_finOnsets = imag_finOnsets[maskImag]
        ftriggers = ftriggers[mask]

        ftriggers[ftriggers == class1] = 0
        ftriggers[ftriggers == class2] = 1

        hSize = int((ftriggers.shape[0]))

        self.trainTriggers = np.zeros((1, hSize), dtype=int)
        self.testTriggers = np.zeros((1, hSize), dtype=int)
        self.trainTriggersAllClasses = np.zeros((1, ftriggersAllClasses.shape[0]), dtype=int)
        self.trainOnsetsAllClasses = np.zeros((1, train_fonsetsAllClasses.shape[0]), dtype=int)
        self.trainFinOnsetsAllClasses = np.zeros((1, train_finOnsetsAllClasses.shape[0]), dtype=int)
        self.trainTriggers[0] = ftriggers
        self.testTriggers[0] = ftriggers
        self.trainTriggersAllClasses[0] = ftriggersAllClasses
        self.trainOnsetsAllClasses[0] = train_fonsetsAllClasses
        self.trainFinOnsetsAllClasses[0] = train_finOnsetsAllClasses

        self.trainOnsets = np.zeros((1, hSize), dtype=int)
        self.finTrainOnsets = np.zeros((1, hSize), dtype=int)
        self.testOnsets = np.zeros((1, hSize), dtype=int)
        self.finTestOnsets = np.zeros((1, hSize), dtype=int)
        self.trainOnsets[0] = train_fonsets
        self.finTrainOnsets[0] = train_finOnsets
        self.testOnsets[0] = test_fonsets
        self.finTestOnsets[0] = test_finOnsets
        return 83


if __name__ == '__main__':
    Fs = 125
    winSize = 3
    step = 0.5  # mandatory
    doAverageReferencing = False
    targetFs = None
    channelSelect = None
    fDiaps = None
    usePCA = False
    numPc = 30
    doHanning = False
    useICA = False
    icaFile = None
    finalClassifier = 'MLP'
    FeatureFunction = featureFuncs.get_spectrum
    DistanceFunction = None  # default: no function (in case your classifier outputs labels directly)
    sensorsMap = None

    workbook = xlsxwriter.Workbook('Results.xlsx')
    worksheet = workbook.add_worksheet()
    subjects = []
    classRates2 = []
    worksheet.write(0, 0, '')
    cell_format1 = workbook.add_format()
    cell_format1.set_bg_color('green')
    cell_format2 = workbook.add_format()
    cell_format2.set_bg_color('yellow')
    cell_format3 = workbook.add_format()
    cell_format3.set_bg_color('orange')
    cell_format4 = workbook.add_format()
    cell_format4.set_bg_color('red')

    genpath = 'V://EEG//Motor//DATA'
    row = 0
    success = 0
    allCount = 0
    print(os.listdir(genpath))
    for subj in os.listdir(genpath):
        pc = mainTest(genpath, subj)
        PortalParams = Params(Fs=Fs, winSize=winSize, step=step, usePCA=usePCA, numPc=numPc,
                              doAverageReferencing=doAverageReferencing, channelSelect=channelSelect, fDiaps=fDiaps,
                              useICA=useICA, icaFile=icaFile, finalClassifier=finalClassifier, doHanning=doHanning,
                              distanceFun=DistanceFunction, featureFun=FeatureFunction, sensorsPosFile=sensorsMap)

        print('Analysing Subject: ' + subj)
        row = row + 1
        PortalWorker = Portal_Classic(PortalParams)
        worksheet.write(row, 0, subj)
        n = 0
        for i in range(0, 5):
            for j in range(1, 5):
                if i < j:
                    print('Training On Subject: ' + subj + ' ' + str(i + 1) + ' vs ' + str(j + 1))
                    worksheet.write(0, n + 1, str(i) + '-' + str(j))
                    pc.setParamsFor2Files(i, j)
                    PortalWorker.train(pc.trainfiles, pc.trainTriggers, pc.trainOnsets, pc.finTrainOnsets,
                                       doBalanceLabels=True)
                    classRates_, confMat_ = PortalWorker.validate(pc.testfiles, pc.testTriggers, pc.testOnsets,
                                                                  pc.finTestOnsets)
                    val1 = int(np.round(100 * classRates_[0]))
                    val2 = int(np.round(100 * classRates_[1]))
                    if (val1 > 75) and (val2 > 75):
                        worksheet.write(row, n + 1, str(val1) + '/' + str(val2), cell_format1)
                    elif (val1 > 60) and (val2 > 60):
                        worksheet.write(row, n + 1, str(val1) + '/' + str(val2), cell_format2)
                    elif (val1 > 50) and (val2 > 50):
                        worksheet.write(row, n + 1, str(val1) + '/' + str(val2), cell_format3)
                    else:
                        worksheet.write(row, n + 1, str(val1) + '/' + str(val2), cell_format4)
                    if (classRates_[0] > 0.5) and (classRates_[1] > 0.5):
                        classRates2.append(classRates_[0])
                        classRates2.append(classRates_[1])
                        success = success + 1
                    n = n + 1
                    allCount = allCount + 1
    row = row + 2
    average = 100 * np.mean(np.array(classRates2))
    worksheet.write(row, 0, 'Valid: ' + str(success / float(allCount)))
    worksheet.write(row + 1, 0, 'Acc: ' + str(average))
    workbook.close()
