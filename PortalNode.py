from PortalParams import Params
import os
import pickle
from PortalClassic import *
from mne.io import RawArray, read_raw_edf
from adaptivePreprocessing import autoSelectComps
from scipy.io import loadmat


class PortalProcessorNode(ProcessorNode):
    def __init__(self, PortalSettings, onsetsPath = None, triggersPath = None, icaPath = None, sensorLocPath = None, mode = 'none', preproc = 'none'):
        #modes: none, train, process
        #preproc: none, adafreq
        mne_info = self.traverse_back_and_find('mne_info')
        Fs = mne_info['sfreq']
        winSize = PortalSettings.winSize
        step = PortalSettings.step
        usePCA = True
        numPC = PortalSettings.numPCAcomps
        if (numPC<=0):
            usePCA = False
        useICA = False
        if not (icaPath is None):
            useICA = True
        finalClassifier = PortalSettings.ClassifierAlgFun
        DistanceFunction = PortalSettings.FeatureDistanceFun
        FeatureFunction = PortalSettings.FeatureExtractionFun

        PortalParams = Params(Fs=Fs, winSize=winSize, step=step, usePCA=usePCA, numPc=numPC,
                              doAverageReferencing=False, channelSelect=None, fDiaps=None,
                              useICA=useICA, icaFile=icaPath, finalClassifier=finalClassifier, doHanning=False,
                              distanceFun=DistanceFunction, featureFun=FeatureFunction, sensorsPosFile=sensorLocPath, preproc=preproc)
        self.PortalParams = PortalParams
        self.PortalWorker = Portal_Classic(PortalParams)
        self.mode = mode

        lab_file = open(triggersPath, "r")
        ftriggers = list(map(int, lab_file.readlines()))
        ftriggers = np.array(ftriggers)
        lab_file.close()
        train_onset_file = open(onsetsPath, "r")
        train_onset_end_file = open(self.traindir + "onsets_end.txt", "r")
        train_fonsets = list(map(float, train_onset_file.readlines()))
        train_fonsets = np.array(train_fonsets)
        train_onset_file.close()
        train_fonsets_end = list(map(float, train_onset_end_file.readlines()))
        train_fonsets_end = np.array(train_fonsets_end)
        train_onset_end_file.close()

        train_fonsets = np.round(train_fonsets * Fs).astype(int)
        train_finOnsets = np.round(train_fonsets_end * Fs).astype(int)
        train_fonsets = train_fonsets
        ftriggersAllClasses = ftriggers
        train_finOnsetsAllClasses = train_finOnsets
        train_fonsetsAllClasses = train_fonsets
        hSize = int((ftriggers.shape[0]))

        self.trainTriggers = np.zeros((1, hSize), dtype=int)
        self.trainTriggersAllClasses = np.zeros((1, ftriggersAllClasses.shape[0]), dtype=int)
        self.trainOnsetsAllClasses = np.zeros((1, train_fonsetsAllClasses.shape[0]), dtype=int)
        self.trainFinOnsetsAllClasses = np.zeros((1, train_finOnsetsAllClasses.shape[0]), dtype=int)
        self.trainTriggers[0] = ftriggers
        self.trainTriggersAllClasses[0] = ftriggersAllClasses
        self.trainOnsetsAllClasses[0] = train_fonsetsAllClasses
        self.trainFinOnsetsAllClasses[0] = train_finOnsetsAllClasses
        self.trainOnsets = np.zeros((1, hSize), dtype=int)
        self.finTrainOnsets = np.zeros((1, hSize), dtype=int)
        self.trainOnsets[0] = train_fonsets
        self.finTrainOnsets[0] = train_finOnsets

    def _loadFromFile(self,basedir, pickleFile, mode='none'):
        path_to_pkl = basedir+'//'+pickleFile
        if (os.path.isfile(path_to_pkl)):
            with open(path_to_pkl, 'rb') as workerFile:
                self.PortalWorker = pickle.load(workerFile)
                self.mode = mode
        else:
            FileExistsError('Pickle File with Portal Object not found: ' + pickleFile)

    def _saveWorkerToFile(self, basedir, pickleFile):
        path_to_pkl = basedir + '//' + pickleFile
        with open(path_to_pkl, 'wb') as workerFile:
            pickle.dump(self.PortalWorker, workerFile, pickle.HIGHEST_PROTOCOL)

    def _initialize(self):
        mne_info = self.traverse_back_and_find('mne_info')
        Fs = mne_info['sfreq']
        self.PortalParams.Fs = Fs
        self.PortalWorker = Portal_Classic(self.PortalParams)

    def _update(self):
        mne_info = self.traverse_back_and_find('mne_info')
        input_array = self.input_node.output
        if self.mode=='train':
            raw_array = RawArray(input_array, mne_info, verbose='ERROR')
            raw_array.pick_types(eeg=True, meg=False, stim=False, exclude='bads')
            if (self.PortalWorker.params.preproc=='adafreq'):
                self.PortalWorker.params.setWinSizeAndStep(winSize=2, step=1)
                [comps, diaps] = autoSelectComps(self.PortalWorker.params, list([raw_array]),
                                                                       self.trainTriggersAllClasses,
                                                                       self.trainOnsetsAllClasses,
                                                                       self.trainFinOnsetsAllClasses,
                                                                       compThresh=0.65, minCompsNum=5)
                self.PortalWorker.params.winSize=self.PortalParams.winSize
                self.PortalWorker.params.step = self.PortalParams.step
                self.PortalWorker.params.channelSelect = comps
                self.PortalWorker.params.fDiaps = diaps
            self.PortalWorker.train(list([raw_array]), self.trainTriggers, self.trainOnsets, self.finTrainOnsets,
                               doBalanceLabels=True)
        if (self.mode=='test'):
            self.output = self.PortalWorker.processChunk(input_array)
