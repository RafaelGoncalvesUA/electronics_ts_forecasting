from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class CustomPCA:
    def __init__(self, var_treshold=0.95):
        self.treshold = var_treshold

    def apply(self, data, targets):
        X = data.drop(targets, axis=1)
        Y = data[targets]

        pca = PCA()
        pca.fit(X)
        n_components_to_keep = np.where(np.cumsum(pca.explained_variance_ratio_) > self.treshold)[0][0]

        pca = PCA(n_components=n_components_to_keep)
        X = pd.DataFrame(pca.fit_transform(X), columns=[f'PC{i}' for i in range(n_components_to_keep)])

        data = pd.concat([X, Y], axis=1)

        return data