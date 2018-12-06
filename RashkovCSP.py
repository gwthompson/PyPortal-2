import numpy as np
from mne.decoding.csp import _ajd_pham

class RashkovCSP():
    def __init__(self, Fs, numPatterns=5):
        self.numPatterns = numPatterns

    def get_csp_filters_twoclass(self, R1, R2, n=5, csp_type="RG"):
        R2_inv = np.linalg.inv(R2)
        C = np.dot(R1, R2_inv)
        w, v = np.linalg.eig(C)
        w = np.asarray(w)
        w_abs = np.abs(w)
        # w = w_abs
        max_filters = []
        min_filters = []
        if csp_type == "STD":
            w_sort = np.sort(w)
            max_filters = [v[:, np.where(w == k)[0][0]] for k in w_sort[-n:]]
            min_filters = [v[:, np.where(w == k)[0][0]] for k in w_sort[:n]]
        elif csp_type == "RG":
            max_div = []
            min_div = []
            for ind, k in enumerate(w):
                vect = v[:, ind]
                div = np.dot(np.dot(np.transpose(vect), R1), vect) / np.dot(np.dot(np.transpose(vect), R2), vect)
                max_div.append(div)
                min_div.append(div)
                div_idx = len(max_div) - 1
                while max_div[div_idx] > max_div[div_idx - 1] and div_idx > 0:
                    t = max_div[div_idx - 1]
                    max_div[div_idx - 1] = max_div[div_idx]
                    max_div[div_idx] = t
                    div_idx -= 1
                max_filters.insert(div_idx, vect)
                if len(max_div) > n:
                    max_div = max_div[:n]
                    max_filters = max_filters[:n]
                div_idx = len(min_div) - 1
                while min_div[div_idx] < min_div[div_idx - 1] and div_idx > 0:
                    t = min_div[div_idx - 1]
                    min_div[div_idx - 1] = min_div[div_idx]
                    min_div[div_idx] = t
                    div_idx -= 1
                min_filters.insert(div_idx, vect)
                if len(min_div) > n:
                    min_div = min_div[:n]
                    min_filters = min_filters[:n]
                print('max', max_div)
            print('min', min_div)
        self.csp_filters = min_filters + max_filters

    def get_csp_filters_multiclass(self, ClassData, n=5, csp_type="RG"):
        C = np.zeros((len(ClassData), ClassData[0].shape[1], ClassData[1].shape[1]))
        sample_weights = list()
        numClasses = len(ClassData)
        numChannels = ClassData[0].shape[1]
        for i in range(numClasses):
            class_ = np.transpose(ClassData[i], [1, 0, 2])
            class_ = class_.reshape(numChannels, -1)
            C[i] = np.cov(class_)
            weight = ClassData[i].shape[1]
            sample_weights.append(weight)

        w, v = _ajd_pham(C)
        mean_cov = np.average(C, axis=0, weights=sample_weights)

        w_, v_ = np.linalg.eig(C[0])
        w = np.asarray(w)
        max_filters = []
        min_filters = []
        if csp_type == "STD":
            w_sort = np.sort(w)
            max_filters = [v[:, np.where(w == k)[0][0]] for k in w_sort[-n:]]
            min_filters = [v[:, np.where(w == k)[0][0]] for k in w_sort[:n]]
        elif csp_type == "RG":
            max_div = []
            min_div = []
            for ind, k in enumerate(w):
                vect = v[:, ind]
                div = np.dot(np.dot(np.transpose(vect), R1), vect) / np.dot(np.dot(np.transpose(vect), R2), vect)
                max_div.append(div)
                min_div.append(div)
                div_idx = len(max_div) - 1
                while max_div[div_idx] > max_div[div_idx - 1] and div_idx > 0:
                    t = max_div[div_idx - 1]
                    max_div[div_idx - 1] = max_div[div_idx]
                    max_div[div_idx] = t
                    div_idx -= 1
                max_filters.insert(div_idx, vect)
                if len(max_div) > n:
                    max_div = max_div[:n]
                    max_filters = max_filters[:n]
                div_idx = len(min_div) - 1
                while min_div[div_idx] < min_div[div_idx - 1] and div_idx > 0:
                    t = min_div[div_idx - 1]
                    min_div[div_idx - 1] = min_div[div_idx]
                    min_div[div_idx] = t
                    div_idx -= 1
                min_filters.insert(div_idx, vect)
                if len(min_div) > n:
                    min_div = min_div[:n]
                    min_filters = min_filters[:n]
                print('max', max_div)
            print('min', min_div)
        self.csp_filters = min_filters + max_filters

    def process(self, data):
        data_inv = np.linalg.inv(data)
        C = np.dot(data, data_inv)
        res = np.array([np.dot(np.dot(filt, C), np.transpose(filt)) for filt in self.csp_filters])
        return res