import time
import numpy as np
import nolds
import scipy.spatial.distance as dist

# Signals Features Extractor class
# Libs:
# numpy
# scipy
# nolds (pip install nolds)
#
# Inputs:
# S         - signals matrix (channels x time)
#
# Methods:
# PFD           - Petrosian fractal dimension
# HFD           - Higushu fractal dimension
# Hurst         - Hurst exponent
# Shannon       - Shannon entropy
# ApEn          - Approximate entropy
# SampEn        - Sample entropy
# MultiSampEn   - Multiscale sample entropy
# SVDEn         - SVD entropy
# Fisher        - Fisher information
# Hjorth        - Hjorth parameters
# DFA           - Detrended fluctuation analysis

class FeatureExtractor():
    def methods(self):
        return ['PFD', 'HFD', 'Hurst', 'Shannon', 'ApEn', 'SampEn', 'MultiSampEn', 'SVDEn', 'Fisher', 'Hjorth', 'DFA']


    def extract(self, S, methods,
                hfd_kmax=5,
                apen_m=10, apen_r=3,
                sampen_m=10, sampen_r=3,
                msampen_delta=3,
                svden_tau=1, svden_dim=2,
                fisher_tau=1, fisher_dim=2,
                dfa_order=3):
        S = self._checkmat(S)
        features = np.empty((S.shape[0], len(methods)+2))
        for i, method in enumerate(methods):
            if method == 'PFD':
                features[:, i] = self.pfd(S)
            elif method == 'HFD':
                features[:, i] = self.hfd(S, kmax=hfd_kmax)
            elif method == 'Hurst':
                features[:, i] = self.hurst(S)
            elif method == 'Shannon':
                features[:, i] = self.shannon(S)
            elif method == 'ApEn':
                features[:, i] = self.ap_entropy(S, dim=apen_m, r=apen_r)
            elif method == 'SampEn':
                features[:, i] = self.samp_entropy(S, dim=sampen_m, r=sampen_r)
            elif method == 'MultiSampEn':
                features[:, i] = self.multi_samp_entropy(S, dim=sampen_m, r=sampen_r, delta=msampen_delta)
            elif method == 'SVDEn':
                features[:, i] = self.svd_entropy(S, dim=svden_dim, delta=svden_tau)
            elif method == 'Fisher':
                features[:, i] = self.fisher(S, dim=fisher_dim, delta=fisher_tau)
            elif method == 'Hjorth':
                features[:, i:i+3] = self.hjorth(S)
            elif method == 'DFA':
                features[:, i+2] = self.dfa(S, order=dfa_order)
            else:
                print('There is no method {}'.format(method))
        return features


    def timetest(self, S, *methods,
                 nt=100,
                 hfd_kmax=5,
                 apen_m=10, apen_r=3,
                 sampen_m=10, sampen_r=3,
                 svden_tau=1, svden_dim=2,
                 fisher_tau=1, fisher_dim=2,
                 dfa_order=3):
        print('Time test init..')
        print('Number of iterations: {}'.format(nt))
        if methods[0] == 'All':
            methods = self.methods()
        for method in methods:
            t = time.time()
            for i in range(nt):
                self.extract(S, method,
                             hfd_kmax=hfd_kmax, apen_m=apen_m, apen_r=apen_r, sampen_m=sampen_m, sampen_r=sampen_r, svden_tau=svden_tau, svden_dim=svden_dim, fisher_tau=fisher_tau, fisher_dim=fisher_dim, dfa_order=dfa_order)
            t = time.time() - t
            print('\nMethod {}'.format(method))
            print('Elapsed time: {} sec.'.format(t))


    def _checkmat(self, m):
        if m.ndim == 1:
            m = np.reshape(m, (1, len(m)))
        return m


    def _toseqs(self, S, dim, delta=1):
        n = S.shape[1] - (dim - 1) * delta
        x = np.empty((S.shape[0], dim, n))
        for i in range(dim):
            x[:, i, :] = S[:, i * delta : i * delta + n]
        return x


    def pfd(self, S):
        S = self._checkmat(S)
        diff = np.diff(S, axis=1)
        prod = diff[:, 1:-1] * diff[:, 0:-2]
        n_delta = np.sum(prod < 0, axis=1)
        n = S.shape[1]
        fd_petrosian = np.log(n) / (np.log(n) + np.log(n / (n + 0.4 * n_delta)))
        return fd_petrosian


    def hfd(self, S, kmax=5):
        S = self._checkmat(S)
        N = S.shape

        km_idxs = np.triu_indices(kmax - 1)
        km_idxs = kmax - np.flipud(np.column_stack(km_idxs)) - 1
        km_idxs[:, 1] -= 1

        Lk = np.empty((N[0], kmax - 1))
        X = np.ones((N[0], kmax - 1, 2))
        for k in range(1, kmax):
            lk = 0
            for m in range(0, k):
                idxs = np.arange(1, int(np.floor((N[1] - m) / k)))
                Lmk = np.sum(np.abs(S[:, m + idxs * k] - S[:, m + k * (idxs - 1)]), axis=1)
                Lmk = (Lmk * (N[1] - 1) / (((N[1] - m) / k) * k)) / k
                lk += Lmk
            lk = np.log(lk / k)
            lk[lk == -np.inf] = 0
            Lk[:, k - 1] = lk
            X[:, k - 1, 0] = np.log(np.ones(N[0]) / k)

        fd_higushi = np.zeros(N[0])
        for i in range(N[0]):
            (p, r1, r2, s) = np.linalg.lstsq(X[i, :, :], Lk[i, :], rcond=None)
            fd_higushi[i] = p[0]
        return fd_higushi


    def hurst(self, S):
        if S.ndim > 1:
            return np.array([self.hurst(signal) for signal in S])
        else:
            return nolds.hurst_rs(S)


    def shannon(self, S):
        if S.ndim > 1:
            return np.array([self.shannon(signal) for signal in S])
        else:
            p = np.histogram(S, bins='auto')[0]
            p = p / np.sum(p)
            p[p == 0] = 1
            return -np.sum(p * np.log2(p))


    def ap_entropy(self, S, dim=10, r=3):
        S = self._checkmat(S)
        s = S.shape

        def get_phi(mn):
            N = s[1] - mn + 1
            x = self._toseqs(S, mn)
            phi = np.zeros(s[0])
            for i in range(s[0]):
                dm = dist.squareform(dist.pdist(np.transpose(x[i, :, :]), 'chebyshev'))
                Cim = np.sum(dm <= r, axis=1) / N
                phi[i] = np.sum(np.log(Cim)) / N
            return phi

        return get_phi(dim) - get_phi(dim + 1)


    def samp_entropy(self, S, dim=10, r=3):
        return self.multi_samp_entropy(S, dim=dim, r=r, delta=1)


    def multi_samp_entropy(self, S, dim=10, r=3, delta=3):
        S = self._checkmat(S)
        s = S.shape

        def get_n(mn):
            x = self._toseqs(S, mn, delta)
            nv = np.zeros(s[0])
            for i in range(s[0]):
                dm = dist.squareform(dist.pdist(np.transpose(x[i, :, :]), 'chebyshev'))
                nv[i] = np.sum(dm < r) - len(dm) * int(r > 0)
            return nv

        A = get_n(dim + 1)
        B = get_n(dim)
        B[B == 0] = 1e-8
        C = A / B
        C[C == 0] = 1
        C[C == np.NaN] = 1
        return -np.log(C)


    def svd_entropy(self, S, dim=2, delta=1):
        if S.ndim > 1:
            return np.array([self.svd_entropy(signal, dim, delta) for signal in S])
        else:
            S = self._checkmat(S)
            mat = self._toseqs(S, dim, delta)[0].T
            W = np.linalg.svd(mat, compute_uv=False)
            W /= np.sum(W)
            entropy_svd = -np.sum(W * np.log2(W))
            return entropy_svd


    def fisher(self, S, dim=2, delta=1):
        if S.ndim > 1:
            return np.array([self.fisher(signal, dim, delta) for signal in S])
        else:
            S = self._checkmat(S)
            mat = self._toseqs(S, dim, delta)[0].T
            W = np.linalg.svd(mat, compute_uv=False)
            W /= sum(W)
            FI_v = (W[1:] - W[:-1])**2 / W[:-1]
            fisher_info = np.sum(FI_v)
            return fisher_info


    def hjorth(self, S):
        S = self._checkmat(S)
        var = np.var(S, axis=1, ddof=1)
        activity = var

        def dsdt(s):
            s_diff = np.diff(s, 1, axis=1)
            return np.pad(s_diff, [(0, 0), (1, 0)], mode='constant', constant_values=0)

        dS = dsdt(S)
        dvar = np.var(dS, axis=1, ddof=1)
        mobility = np.sqrt(dvar / var)

        ddS = dsdt(dS)
        complexity = np.sqrt(np.var(ddS, axis=1, ddof=1) * var) / dvar
        return np.transpose(np.vstack((activity, mobility, complexity)))


    def dfa(self, S, n=None, order=1):
        S = self._checkmat(S)
        X = np.cumsum(S - np.mean(S, axis=1, keepdims=True), axis=1)
        if n is None:
            n = int(np.sqrt(S.shape[1]))
        ids = np.linspace(0, X.shape[1], n + 1).astype(int)
        F = 0
        for i in range(n):
            segment = X[:, ids[i]:ids[i + 1]]
            arg = np.arange(1, segment.shape[1] + 1)
            coeffs = np.polynomial.polynomial.polyfit(arg, np.transpose(segment), order)
            fits = np.polynomial.polynomial.polyval(arg, coeffs)
            F += np.sum((segment - fits) ** 2, axis=1)
        if S.shape[0] == 1:
            F = F[0]
        return np.sqrt(F / S.shape[1])