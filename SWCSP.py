import numpy as np
from scipy.linalg import eigh
from scipy.fftpack import fft, fftfreq, ifft

class SWCSP():
    def __init__(self, Fs, numPatterns=5):
        self.pp = 0
        self.qp = 1
        self.p = self.pp + self.qp
        self.q = self.qp
        self.numPatterns = numPatterns
        self.lowFreq = 1
        self.highFreq = 35
        self.steps = 3
        self.Fs = Fs
        self.alpha=[]
        self.W = []

    def train(self,trainData):
        V = []
        F=[]
        tlen = trainData[0].shape[2]
        freqs = np.arange(tlen - 1) * self.Fs / tlen
        maskFreqs = (freqs >= self.lowFreq) & (freqs <= self.highFreq)
        bands = np.nonzero(maskFreqs)
        for c in range(2):
            classData = trainData[c]
            nTrials = classData.shape[0]
            chans = classData.shape[1]
            fftTrainData = np.zeros(classData.shape)
            for nc in range(chans):
                for nt in range(nTrials):
                    fftTrainData[nt,nc,:] = np.real(fft(classData[nt,nc,:]))
            F_ = np.zeros((chans,chans,np.max(bands)+1,nTrials))
            for _, k in np.ndenumerate(bands):
                for t in range(nTrials):
                    vec = fftTrainData[t,:,k]
                    vec = np.expand_dims(vec,0)
                    F_[:,:,k,t] = 2*(np.multiply(np.transpose(vec),vec))
            V.append(np.mean(F_,3))
            F.append(F_)

        J=1
        alpha=np.zeros((self.numPatterns*2, np.max(bands)+1))
        alpha[:,bands]=1.0

        for step in range(self.steps):
            W = np.zeros((J, chans, 2 * self.numPatterns))
            #P = np.zeros((J, chans, 2 * self.numPatterns))
            lbda = np.zeros((J,2))
            for j in range(J):
                Sigma = [np.zeros((chans,chans))]*2
                for c in range(2):
                    for _,b in np.ndenumerate(bands):
                        Sigma[c] = Sigma[c] + alpha[j,b]*V[c][:,:,b]
                DD, VV = eigh(Sigma[0],Sigma[0] + Sigma[1],eigvals_only=False)
                W[j,:,0:self.numPatterns]= VV[:, :self.numPatterns]
                W[j,:,self.numPatterns:] = VV[:, -self.numPatterns:]
                #iVV = np.linalg.inv(VV)
                #P[j, :, 0:self.numPatterns] = iVV[:, :self.numPatterns]
                #P[j, :, self.numPatterns:] = iVV[:, -self.numPatterns:]
                lbda[j,0] = DD[0]
                lbda[j,1] = DD[-1]
            indMin = np.argmin(lbda[:,0])
            indMax = np.argmin(lbda[:,1])
            Wcat = np.concatenate((W[indMin,:,0:self.numPatterns],W[indMax,:,self.numPatterns:]),axis=1)
            #Pcat = np.concatenate((P[indMin,:,0:self.numPatterns],P[indMax,:,self.numPatterns:]),axis=1)
            J = 2 * self.numPatterns
            for j in range(J):
                print("Step {}, pattern {}".format(step,j))
                w = np.expand_dims(Wcat[:,j],0)
                wt = np.transpose(w)
                mu_s=[]
                var_s=[]
                alpha_opt = np.zeros((2, np.max(bands) + 1))
                alpha_tmp = np.zeros((2, np.max(bands) + 1))
                for c in range(2):
                    Fclass = F[c]
                    nTrials = trainData[c].shape[0]
                    s = np.zeros((nTrials,np.max(bands)+1))
                    for _, b in np.ndenumerate(bands):
                        for t in range(nTrials):
                            tmp=np.dot(w,Fclass[:,:,b,t])
                            s[t, b] = np.dot(tmp,wt)
                    mu_s.append(np.mean(s,axis=0))
                    var_s.append(np.var(s,axis=0))
                for c in range(2):
                    for _, b in np.ndenumerate(bands):
                        val_opt = (mu_s[c][b]-mu_s[1-c][b]) / (var_s[0][b] + var_s[1][b])
                        val_tmp = np.power(val_opt,self.q) * np.power(freqs[b]*(mu_s[0][b]+mu_s[1][b]/2.0),self.p)
                        alpha_opt[c][b]=np.max(np.array([0.0, val_opt]))
                        alpha_tmp[c][b]=val_tmp
                alpha[j,:] = np.maximum(alpha_tmp[0], alpha_tmp[1])
                alpha[j,:] = alpha[j,:]/np.sum(alpha[j,:])
        alpha = np.concatenate((alpha, np.zeros((alpha.shape[0], tlen - alpha.shape[1]))), axis=1)
        self.alpha=alpha
        self.W = Wcat

    def process(self, data):
        Wprod = np.dot(np.transpose(data),self.W)
        fftWprod = fft(Wprod)
        afft = np.multiply(self.alpha, np.transpose(fftWprod))
        transf = np.log(np.var(2 * np.real(ifft(afft)), axis=1))
        return transf