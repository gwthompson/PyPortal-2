import numpy as np
import utilities
from scipy.spatial.distance import mahalanobis

def kNear(trainVectors, trainLabels, testVector, params):
    dists = np.zeros((trainVectors.shape[0]))
    numClasses = len(set(trainLabels))
    for j, tfvec in enumerate(trainVectors):
        dists[j] = utilities.eucDist(testVector, tfvec)
    inds = np.argsort(dists)
    numNeighbors = int(np.floor(trainVectors.shape[0]/numClasses/2))
    inds = inds[0:numNeighbors]
    sortedLabs = trainLabels[inds]
    cNear = np.zeros((numClasses))
    for c in range(numClasses):
        cNear[c] = np.sum(sortedLabs == c)
    result = np.argsort(cNear)[-1]
    return result

def centroidEucDist(trainVectors, trainLabels, testVector, params):
    numClasses = len(set(trainLabels))
    centroids = np.zeros((numClasses, trainVectors.shape[1]))
    dists = np.zeros((numClasses))
    for i in range(numClasses):
        idx = np.where(trainLabels == i)
        centroids[i, :] = np.mean(trainVectors[idx, :], 1)
        dists[i] = utilities.eucDist(centroids[i,:], testVector)
    result = np.argmin(dists)
    return result

def centroidMahDist(trainVectors, trainLabels, testVector, params):
    covMat = np.cov(np.transpose(trainVectors))
    rk = np.linalg.matrix_rank(covMat)
    if (rk<covMat.shape[0]):
        raise ValueError('Cannot calculate Mahalanobis distance. Rank of covMat {} instead of {}'.format(rk,covMat.shape[0]))
    invcov = np.linalg.inv(covMat)
    numClasses = len(set(trainLabels))
    centroids = np.zeros((numClasses, trainVectors.shape[1]))
    dists = np.zeros((numClasses))
    for i in range(numClasses):
        idx = np.where(trainLabels == i)
        centroids[i, :] = np.mean(trainVectors[idx, :], 1)
        dists[i] = mahalanobis(centroids[i,:], testVector,invcov)
    result = np.argmin(dists)
    return result