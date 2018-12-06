from PortalClassic import *
from PortalIndepChan import *
from PortalCustomFreqs import *
from PortalCSP import *
from os import listdir
from os.path import isfile, join
import scipy.io

class mainTest:
    def setParams(self):
        basedir = 'E://Databases//EEG//Visual Search Task//Alekseev//Session1//'
        self.trainfiles = [basedir + f for f in listdir(basedir) if f.endswith("44.edf")]
        event_file = open(basedir + "events.txt", "r")
        events = [ln.split('\t') for ln in event_file.readlines()]
        evtypes = list(map(int, [event[0] for event in events]))
        labels = list(map(int, [event[1] for event in events]))
        onsets = list(map(int, [event[2] for event in events]))
        event_file.close()
        labels = np.array(labels) - 1
        onsets = np.array(onsets) - 1
        searchOnsets=[]
        searchOnsetsFin=[]
        searchLabels = []

        for i in range(len(onsets)):
            if evtypes[i] == 2:
                searchOnsets.append(onsets[i])
                searchLabels.append(labels[i])
                searchOnsetsFin.append(onsets[i+1]-1)

        searchOnsets=np.array(searchOnsets)
        searchLabels = np.array(searchLabels)
        searchOnsetsFin = np.array(searchOnsetsFin)

        class1 = 2
        class2 = 5

        mask = (searchLabels == class1) | (searchLabels == class2)
        fonsets = searchOnsets[mask]
        finonsets = searchOnsetsFin[mask]
        ftriggers = searchLabels[mask]
        ftriggers[ftriggers == class1] = 0
        ftriggers[ftriggers == class2] = 1

        hSize = ftriggers.shape[0] / 2.0
        Fs = 500

        self.trainTriggers = np.zeros((1, int(np.ceil(hSize))), dtype=int)
        self.testTriggers = np.zeros((1, int(np.floor(hSize))), dtype=int)
        self.trainTriggers[0] = ftriggers[0:int(np.ceil(hSize))]
        self.testTriggers[0] = ftriggers[int(np.ceil(hSize)):]

        self.trainOnsets = np.zeros((1, int(np.ceil(hSize))), dtype=int)
        self.finTrainOnsets = np.zeros((1, int(np.ceil(hSize))), dtype=int)
        self.testOnsets = np.zeros((1, int(np.floor(hSize))), dtype=int)
        self.finTestOnsets = np.zeros((1, int(np.floor(hSize))), dtype=int)
        self.trainOnsets[0] = fonsets[0:int(np.ceil(hSize))]
        self.finTrainOnsets[0] = finonsets[0:int(np.ceil(hSize))]
        self.testOnsets[0] = fonsets[int(np.ceil(hSize)):]
        self.finTestOnsets[0] = finonsets[int(np.ceil(hSize)):]
        return 83

    def train(self):
        PortalParams = Params(2,500,range(0,128),5)
        PortalWorker = PortalIndepChan(PortalParams)
        #PortalWorker = Portal_IndepChan(PortalParams)
        #PortalWorker = Portal_CSP(PortalParams)
        PortalWorker.train(self.trainfiles,self.trainTriggers,self.trainOnsets,self.finTrainOnsets)
        classRates = PortalWorker.validate(self.trainfiles,self.testTriggers,self.testOnsets,self.finTestOnsets)
        return classRates

if __name__ == '__main__':
    pc = mainTest()
    ans = pc.setParams()
    classRates = pc.train()