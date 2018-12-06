# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:59:57 2018

@author: User
"""

import numpy as np
import scipy
import statsmodels.api as sm
import sys
import os
sys.path.append(os.getcwd())
import pyeeg3


def singular_values(X, Tau, DE, normalized=True):
    W = scipy.linalg.svd(pyeeg3.embed_seq(X, Tau, DE), compute_uv=False)
    if normalized==True:
        W /= sum(W)
    return W


class Feature:
    def __init__(self, data):
        self.data = data


class FeatureChannelsIndepend(Feature):
    def list_methods(self):
        return ['PFD', 'HFD', 'Hurst', 'ApEn', 'SampEn', 'SVDEn', 'FisherInfo', 'Hjorth', 'DFA', 'UAR']

    def calculate(self, *methods, kmax=5, M_ae=10, R_ae=None, M_se=10, R_se=None,
                  Tau_svd=4, DE_svd=10, Tau_f=4, DE_f=10, L=None, p=None):
        features = []
        for method in methods:
            if method=='PFD':
                features.append(self.pfdim())
            elif method=='HFD':
                features.append(self.hfdim(kmax=kmax))
            elif method=='Hurst':
                features.append(self.hurst())
            elif method=='ApEn':
                features.append(self.ap_entropy(M=M_ae, R=R_ae))
            elif method=='SampEn':
                features.append(self.samp_entropy(M=M_se, R=R_se))
            elif method=='SVDEn':
                features.append(self.svd_entropy(Tau=Tau_svd, DE=DE_svd))
            elif method=='FisherInfo':
                features.append(self.fisher(Tau=Tau_f, DE=DE_f))
            elif method=='Hjorth':
                features.append(self.hjorth())
            elif method=='DFA':
                features.append(self.dfa(L=L))
            elif method=='UAR':
                features.append(self.uar(p=p))
            else:
                print('There is no method {}'.format(method))
        self.feature_vector = np.hstack((features))
        return self.feature_vector

    def pfdim(self):
        self.pfdim_value = np.array([[pyeeg3.pfd(X)] for X in self.data])
        return self.pfdim_value

    def hfdim(self, kmax=5):
        self.hfdim_value = np.array([[pyeeg3.hfd(X, kmax)] for X in self.data])
        return self.hfdim_value

    def hurst(self):
        # Hurst exponent
        self.hurst_value  = np.array([[pyeeg3.hurst(X)] for X in self.data])
        return self.hurst_value

    def ap_entropy(self, M=10, R=None):
        # Approximate entropy
        # R is 1-D array
        # (Mostly, the value of R is defined as 20% - 30% of standard deviation of X))
        if R==None:
            R=0.3*np.std(self.data, axis=1)
        self.ap_entropy_value = np.array([[pyeeg3.ap_entropy(X, M, r)] for X, r in zip(self.data, R)])
        return self.ap_entropy_value

    def samp_entropy(self, M=10, R=None):
        # Sample entropy
        # R is 1-D array
        # (Mostly, the value of R is defined as 20% - 30% of standard deviation of X))
        if R==None:
            R=0.3*np.std(self.data, axis=1)
        self.samp_entropy_value = np.array([[pyeeg3.samp_entropy(X, M, r)] for X, r in zip(self.data, R)])
        return self.samp_entropy_value

    def svd_entropy(self, Tau=4, DE=10):
        # Singular value decomposition entropy
        # Normalized singular values of embedding matrix:
        W = np.array([singular_values(X, Tau, DE) for X in self.data])
        self.svd_entropy_value = np.array([[pyeeg3.svd_entropy(X, Tau, DE, W=w)] for X, w in zip(self.data, W)])
        return self.svd_entropy_value

    def fisher(self, Tau=4, DE=10):
        # Fisher information
        # Normalized singular values of embedding matrix:
        W = np.array([singular_values(X, Tau, DE) for X in self.data])
        self.fisher_value = np.array([[pyeeg3.fisher_info(X, Tau, DE, W=w)] for X, w in zip(self.data, W)])
        return self.fisher_value

    def hjorth(self):
        # Hjorth mobility and complexity
        D = np.array([pyeeg3.first_order_diff(X) for X in self.data])
        self.hjorth_value = np.array([pyeeg3.hjorth(X, d) for X, d in zip(self.data, D)])
        return self.hjorth_value

    def dfa(self, L=None):
        # Detrended Fluctuation Analysis
        self.dfa_value = np.array([[pyeeg3.dfa(X, L=L)] for X in self.data])
        return self.dfa_value

    def uar(self, p=None):
        # Univariate autoregressive process
        AR_params = []
        for X in self.data:
            AR_model = sm.tsa.AR(X)
            AR_res = AR_model.fit(p)
            AR_params.append(AR_res.params)
        self.uar_value = np.vstack((AR_params))
        return self.uar_value


class FeatureChannelsDepend(Feature):
    def var(self, maxlags=None):
        # Vector autoregressive process
        VAR_model = sm.tsa.VAR(self.data.T)
        VAR_res = VAR_model.fit(maxlags=maxlags)
        vector = VAR_res.params
        self.var_value = vector.T
        return self.var_value

