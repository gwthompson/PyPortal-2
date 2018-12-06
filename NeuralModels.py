import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsummary import summary

import numpy as np

def dataloader(eegs, labels):
    for e, l in zip(eegs, labels):
        yield e, l



class GenericModel(nn.Module):
    def __init__(self, numClasses,lr):
        super(GenericModel, self).__init__()
        self.numClasses = numClasses
        self.lr = lr

    def trainModel(self, trainData, labels, batchSize, numEpochs):
        numSamples = len(labels)
        numBatches = int(np.floor(numSamples / batchSize))
        eegs=[]
        lbls=[]
        inds = np.random.permutation(labels.shape[0])
        sp=0
        for i in range(numBatches):
            curInds = inds[sp:sp + batchSize]
            eegs.append(trainData[curInds, :, :])
            lbls.append(labels[curInds])
            sp = sp + batchSize

        opt = optim.Adam(self.cuda().parameters(),self.lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(numEpochs):
            gen = dataloader(eegs, lbls)
            for (eegs_, y_) in gen:
                eeg = Variable(torch.FloatTensor(eegs_).squeeze()).cuda()
                y = Variable(torch.LongTensor(y_))
                onehot = torch.FloatTensor(batchSize, self.numClasses)
                onehot.zero_()
                onehot.scatter_(1, y.view(batchSize, 1), 1)
                classOut = self.cuda()(eeg)
                classLoss = criterion(classOut, y.cuda())
                opt.zero_grad()
                classLoss.backward(retain_graph=True)
                opt.step()
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, numEpochs, classLoss.item()))

    def testModel(self, testVec):
        eeg = Variable(torch.FloatTensor(testVec).squeeze()).cuda()
        if (np.ndim(testVec)==3):
            eeg = eeg.view(1, eeg.size(0), eeg.size(1), eeg.size(2))
        if (np.ndim(testVec) == 2):
            eeg = eeg.view(1, eeg.size(0), eeg.size(1))
        classOut = self.cuda()(eeg)
        return np.argmax(classOut.cpu().data.numpy())


class BasicConvModel(GenericModel):
    def __init__(self, numClasses,learning_rate=0.0001):
        super(BasicConvModel, self).__init__(numClasses,learning_rate)

        self.modelConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        print('CONV part summary')
        summary(self.modelConv.cuda(),(1,128,128))

        self.modelDense = nn.Sequential(
            nn.Linear(1024, 500),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(0.5),
            nn.Linear(500,self.numClasses),
            nn.Softmax()
        )

        print('Dense part summary')
        summary(self.modelDense.cuda(), (1, 1024))

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        y1 = self.modelConv(x)
        y1 = y1.view(y1.size(0), -1)
        y = self.modelDense(y1)
        return y

class TinyConvModel(GenericModel):
    def __init__(self, numClasses,learning_rate=0.0001):
        super(TinyConvModel, self).__init__(numClasses,learning_rate)

        self.modelConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
        )
        print('CONV part summary')
        summary(self.modelConv.cuda(),(1,128,128))

        self.modelDense = nn.Sequential(
            nn.Linear(576, 100),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(0.5),
            nn.Linear(100,self.numClasses),
            nn.Softmax()
        )

        print('Dense part summary')
        summary(self.modelDense.cuda(), (1, 576))

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        y1 = self.modelConv(x)
        y1 = y1.view(y1.size(0), -1)
        y = self.modelDense(y1)
        return y

class FreqImageEEGEncoder(GenericModel):
    def __init__(self, numClasses,learning_rate=0.0001):
        super(FreqImageEEGEncoder, self).__init__(numClasses,learning_rate)

        self.modelConv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3, 3), stride=1, padding=0,dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,4)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4))
        )

        print('CONV part summary')
        summary(self.modelConv.cuda(), (3, 64, 64))

        self.modelDense = nn.Sequential(
            nn.Linear(64, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50,2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(3),x.size(1),x.size(2))
        x = self.modelConv(x)
        #print(x.size())
        y1 = x.view(x.size(0),-1)
        y2 = self.modelDense(y1)
        return y2

class FreqImageEEGEncoder2(GenericModel):
    def __init__(self, numClasses,learning_rate=0.0001):
        super(FreqImageEEGEncoder2, self).__init__(numClasses,learning_rate)

        self.modelConv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3, 3), stride=1, padding=0,dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4))
        )

        print('CONV part summary')
        summary(self.modelConv.cuda(), (3, 64, 64))

        self.modelDense = nn.Sequential(
            nn.Linear(64, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50,2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(3),x.size(1),x.size(2))
        x = self.modelConv(x)
        #print(x.size())
        y1 = x.view(x.size(0),-1)
        y2 = self.modelDense(y1)
        return y2

class Conv1DModel(GenericModel):
    def __init__(self, numClasses,learning_rate=0.0001):
        super(Conv1DModel, self).__init__(numClasses,learning_rate)

        self.modelConv = nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=64,kernel_size=13, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            # nn.BatchNorm1d(num_features=64),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # nn.BatchNorm1d(num_features=64),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # nn.BatchNorm1d(num_features=64),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        print('CONV part summary')
        summary(self.modelConv.cuda(), (128, 1500))

        self.modelDense = nn.Sequential(
            nn.Linear(256, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50,2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1),x.size(2))
        x = self.modelConv(x)
        #print(x.size())
        y1 = x.view(x.size(0),-1)
        y2 = self.modelDense(y1)
        return y2