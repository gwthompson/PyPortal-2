import scipy.io
import os

class Params:
    def __init__(self, winSize, step, Fs, doAverageReferencing = False, channelSelect = None, usePCA = True, numPc=5, doHanning=False, useICA=False, icaFile = 'ICA_tmp.mat', finalClassifier = None, fDiaps=None, targetFs = None,
                 featureFun = None, distanceFun = None, sensorsPosFile = None, preproc = 'none'):
        self.rawFs = Fs
        self.Fs = Fs
        if not (targetFs is None):
            self.Fs = targetFs
        self.winSize = round(winSize*self.Fs)
        self.step = round(step*self.Fs)
        self.numPC = numPc
        self.doHanning = doHanning
        self.lowFreq = 1
        self.highFreq = 35
        self.channelSelect = []
        self.icaFile = icaFile
        self.useICA = useICA
        self.usePCA = usePCA
        self.finalClassifier = finalClassifier
        self.fDiaps = fDiaps
        self.doAverageReferencing = doAverageReferencing
        self.featureFun = featureFun
        self.distanceFun = distanceFun
        self.sensorsFile = sensorsPosFile
        self.preproc=preproc #toDo: link autoSelectComps here
        if not ((channelSelect is None) or (channelSelect==[])):
            self.channelSelect = channelSelect

    def setWinSizeAndStep(self, winSize, step):
        self.winSize = round(winSize * self.Fs)
        self.step = round(step * self.Fs)

class TrainResult:
    def __init__(self):
        self.mean=[]
        self.std=[]
        self.pcaOp = []
        self.model=[]
        self.cspOp = []
        self.finalOp=[]
        self.trainTransformedVecs=[]
        self.trainLabels=[]
        self.comps=[]
        self.diaps=[]
