from google.protobuf import text_format
import numpy as np
import scipy.io
from SWCSP import SWCSP
import time
print('aaa \n\n\n aaa')

a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
b = np.array([0,0,0,0,1,1,0,0]) > 0

a[b]=0


s1=np.random.randn(100, 128)
td = dict()
td['T'] = s1
scipy.io.savemat('C://Projects//PyPortal//T.mat', td)

s1=np.random.randn(100, 128, 500)
s2=np.random.randn(125, 128, 500)
S=[]
S.append(s1)
S.append(s2)

cspWorker = SWCSP(500)
cur = time.time()
cspWorker.train(S)
print("TrainTime: {} sec".format(time.time()-cur))
cur = time.time()
for i in range(100):
    cspWorker.process(np.squeeze(s1[1:7,:,:]))
elapsed = (float(time.time()-cur))/100.0*1000
print("ProcessTime: {} ms".format(elapsed))
aaa=1
