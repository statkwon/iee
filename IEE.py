import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import is_regressor
from sklearn.inspection import partial_dependence

class Explainer:
    def __init__(self, model):
        self.model = model
        self.type = 'r' if is_regressor(model) else 'c'
    
    def independent_conditional_expectation(self, X, fs, centered):
        ice_values = []
        for i in range(X.shape[1]):
            pdf = partial_dependence(self.model, X, [i], grid_resolution=fs, kind='individual')
            if centered:
                pdf['individual'][0] -= pdf['individual'][0][:, 0].reshape(-1, 1)
            ice_values.append({'values':pdf['values'][0], 'individual':pdf['individual']})
        return ice_values
    
    def __call__(self, X, fs=100, centered=True, scaled=True):
        self.n, self.p = X.shape
        self.ice_values = self.independent_conditional_expectation(X, fs, centered)
        if self.type == 'r':
            vj = []
            for j in range(self.p):
                t = self.ice_values[j]['values']
                s = self.ice_values[j]['individual'][0]
                h = t.size // 2 + 1
                g = 1 / (np.arange(1, h + 1) ** 2)
                vi = []
                for i in range(self.n):
                    ft = (np.fft.fft(s[i]) / t.size)[:h]
                    v = sum(abs(ft) ** 2 / g)
                    vi.append(v)
                vj.append(vi)
            iee_values = np.transpose(vj)
        else:
            vj = []
            for j in range(self.p):
                t = self.ice_values[j]['values']
                s = self.ice_values[j]['individual']
                g = 1 / (np.arange(1, t.size // 2 + 1) ** 2)
                vk = []
                for k in range(len(s)):
                    sk = s[k]
                    vi = []
                    for i in range(self.n):
                        ft = (np.fft.fft(sk[i]) / t.size)[:t.size // 2]
                        v = sum(abs(ft) ** 2 / g)
                        vi.append(v)
                    vk.append(vi)
                vj.append(vk)
            iee_values = np.transpose(vj, [1, 2, 0])
        if scaled:
            iee_values = (iee_values - iee_values.min()) / (iee_values.max() - iee_values.min())
        return iee_values
    
    def ice_plot(self, iee_values, row_idx, class_idx=None, features=None, feature_names=None):
        if class_idx is None:
            class_idx = 0
            if features is None:
                features = range(iee_values.shape[1])
        else:
            if features is None:
                features = range(iee_values[class_idx].shape[1])
        if feature_names is None:
            feature_names = [f'X{i + 1}' for i in features]
        
        fig, ax = plt.subplots()
        for i in features:
            if self.type == 'r':
                label = f'{feature_names[i]}({round(iee_values[row_idx, i], 3)})'
            else:
                label = f'{feature_names[i]}({round(iee_values[class_idx, row_idx, i], 3)})'
            ax.plot(self.ice_values[i]['values'], self.ice_values[i]['individual'][class_idx][row_idx], label=label)
        ax.legend()
        plt.show()