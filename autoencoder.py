__author__ = 'chen'
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def getData():
    labelName = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    trainX = np.ones([11314, 2000], dtype = np.float32)
    trainY = np.ones([11314], dtype = np.int64)
    testX = np.ones([7532, 2000], dtype = np.float32)
    testY = np.ones([7532], dtype = np.int64)

    infile = open('20news-bydate_commonest_train_count2000.txt')
    count = 0
    for line in infile:
        line = line.strip('\n').split(',')
        trainY[count] = int(line[0])
        trainX[count,:] = [(float(x)) for x in line[1:]]
        #trainX[count,:] /= trainX[count,:].sum() #normalization, bad
        count += 1
    print(count)
    infile.close()

    infile = open('20news-bydate_commonest_test_count2000.txt')
    count = 0
    for line in infile:
        line = line.strip('\n').split(',')
        testY[count] = int(line[0])
        testX[count,:] = [(float(x)) for x in line[1:]]
        #testX[count,:] /= testX[count,:].sum() ##normalization, bad
        count += 1
    print(count)
    infile.close()

    #trainX = trainX.reshape([11314,1,2000,1])
    #testX = testX.reshape([7532,1,2000,1])

    return labelName, trainX, trainY, testX, testY


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(2000, 565)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

    def applyMask(self, mask):
        weight = list(self.parameters())[0].data
        weight.mul_(mask)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode = nn.Linear(565, 2000)
        self.classify = nn.Linear(565, 20)
        self.mode = 'decode'

    def forward(self, x):
        if self.mode == 'decode':
            x = self.decode(x)
        elif self.mode == 'classify':
            x = self.classify(x)
        else:
            assert False
        return x


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def applyMask(self, mask):
        self.encoder.applyMask(mask)

    def setMode(self, mode):
        self.decoder.mode = mode
        if mode == 'classify':
            p = list(self.parameters())
            p[0].requires_grad = False #don't update the encoder parameters
            p[1].requires_grad = False
        elif mode == 'decode':
            p = list(self.parameters())
            p[0].requires_grad = True
            p[1].requires_grad = True
        else:
            assert False


labelName, trainX, trainY, testX, testY = getData()
batchsize = 100
batchnum = trainX.shape[0] / batchsize
randgen = np.random.RandomState(11)
randomTestIdx = randgen.choice(7532, 1000, replace=False)
randomTestX = torch.from_numpy(testX[randomTestIdx])
randomTestY = torch.from_numpy(testY[randomTestIdx])
randomTrainIdx = randgen.choice(11314, 1000, replace=False)
randomTrainX = torch.from_numpy(trainX[randomTrainIdx])
randomTrainY = torch.from_numpy(trainY[randomTrainIdx])


mask = torch.from_numpy( sio.loadmat('mask_commonest2000_soft_cond_PEM_565hid_400.mat')['mask'].astype(np.float32).T )

ae = AE()
criterion = nn.MSELoss()
optimizer = optim.SGD(ae.parameters(), lr=0.01, momentum=0.9)#0.001

trainloss = []
testloss = []
for epoch in range(10):
    running_loss = 0.0
    for batch in range(batchnum):
        batchX = torch.from_numpy( trainX[ batch * batchsize: batch * batchsize + batchsize] )
        batchX = Variable(batchX)

        #batchY = torch.from_numpy( trainY[ batch * batchsize: batch * batchsize + batchsize] )
        #batchY = Variable(batchY)

        optimizer.zero_grad()

        #forward, backward, update
        output = ae(batchX)
        loss = criterion(output, batchX / batchX.sum(dim=1).expand_as(batchX))
        loss.backward()
        optimizer.step()
        ae.applyMask(mask)############################################

        # print statistics
        running_loss += loss.data[0]
        if batch % 10 == 9:
            print('[%d, %5d] training batch loss: %.6f' % (epoch+1, batch+1, running_loss / 10))
            trainloss.append( running_loss /10 )
            running_loss = 0.0

            output = ae(Variable(randomTestX))
            loss = criterion(output, Variable(randomTestX) / Variable(randomTestX.sum(dim=1).expand_as(randomTestX)))
            print('random test loss: %.6f' % (loss.data[0]))
            testloss.append(loss.data[0])


# plt.plot(range(len(trainloss)), trainloss)
# plt.plot(range(len(testloss)), testloss)
# plt.show()

print('End of autoencoder!\n\n\n\n\n')
ae.setMode('classify')
criterion = nn.CrossEntropyLoss()
best = 0
for epoch in range(100):
    running_loss = 0.0
    for batch in range(batchnum):
        batchX = torch.from_numpy( trainX[ batch * batchsize: batch * batchsize + batchsize] )
        batchX = Variable(batchX)

        batchY = torch.from_numpy( trainY[ batch * batchsize: batch * batchsize + batchsize] )
        batchY = Variable(batchY)

        optimizer.zero_grad()
        #input = Variable(torch.randn(3,1,10,1))
        #target = Variable(torch.from_numpy(np.array([1,2,3])))

        #forward, backward, update
        output = ae(batchX)
        loss = criterion(output, batchY)
        loss.backward()
        optimizer.step()
        #ae.applyMask(mask)############################################

        # print statistics
        running_loss += loss.data[0]
        if batch % 10 == 9:
            print('[%d, %5d] training batch loss: %.3f' % (epoch+1, batch+1, running_loss / 10))
            running_loss = 0.0

            output = ae(Variable(randomTrainX))
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == randomTrainY).sum()
            print('Accuracy of the network on the random training docs: %.1f %%' % (
                100.0 * correct / predicted.size()[0]))


            output = ae(Variable(randomTestX))
            loss = criterion(output, Variable(randomTestY))
            print('random test loss: %.3f' % (loss.data[0]))

            _, predicted = torch.max(output.data, 1)
            correct = (predicted == randomTestY).sum()

            print('Accuracy of the network on the random test docs: %.1f %%' % (
                100.0 * correct / predicted.size()[0]))

            if 100.0 * correct / predicted.size()[0] > best:
                best = 100.0 * correct / predicted.size()[0]
            print('Best test Accuracy: %.1f %%' % (best))


