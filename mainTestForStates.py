from PortalClassic import *
from PortalIndepChan import *
from PortalConvNet import *
import featureFuncs
from os import listdir
from os.path import isfile, join

trainfiles = [(join('files128//train//', f)) for f in listdir('files128//train//') if isfile(join('files128//train//', f))]
validfiles = [(join('files128//test//', f)) for f in listdir('files128//test//') if isfile(join('files128//test//', f))]
triggers = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]

onsets = [[500*0, 500*30, 500*60, 500*90], [500*0, 500*30, 500*60, 500*90], [500*0, 500*30, 500*60, 500*90]]
finonsets = [[500*30, 500*60, 500*90, 500*120], [500*30, 500*60, 500*90, 500*120], [500*30, 500*60, 500*90, 500*120]]

finonsets = np.array(finonsets)
onsets = np.array(onsets)
triggers=np.array(triggers)

PortalParams = Params(numClasses=4,Fs=500,winSize=2, step=0.5, channelSelect=np.array([0]))
PortalWorker = Portal_Classic(PortalParams,featureFun=featureFuncs.get_spectrum,distanceFun=distanseFuncs.kNear)
PortalWorker.train(trainfiles,triggers,onsets,finonsets,False)
PortalWorker.validate(validfiles,triggers,onsets,finonsets)

#Spectrum: [ 0.70238095  0.98214286  0.74404762  0.83333333]

#HFD:      [ 0.54761905  0.99404762  0.96428571  0.95238095]
#PFD:      [ 0.49404762  0.93452381  0.96428571  0.95833333]
#UAR:      [ 0.46428571  0.98214286  0.875       0.95238095]
#DFA:      [ 0.79761905  0.66071429  0.63690476  0.50595238]
#Hjorth    [ 0.80952381  0.93452381  0.85714286  0.76190476]
#Fischer   [ 0.83928571  0.57142857  0.45238095  0.73214286]

#Sound:
#mfcc               [0.69642857 0.98214286 0.88095238 0.94047619]
#spectral_centroid  [0.76785714 0.96428571 0.86904762 0.92857143]


#PortalParams = Params(4,500,range(0,128),numCSP=5)
# PortalWorker = Portal_SWCSP(PortalParams)
# PortalWorker.train(trainfiles,triggers,onsets,finonsets)
# PortalWorker.validate(validfiles,triggers,onsets,finonsets)