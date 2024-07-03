from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class CustomPCA:
    def __init__(self, var_treshold=0.95):
        self.treshold = var_treshold

    def fit(self, df, to_forecast, out_steps):
        columns_to_keep = [f"{to_forecast}_future{i}" for i in range(1, out_steps+1)] + ['Hour', 'Day Type']
        self.to_forecast = to_forecast
        self.out_steps = out_steps

        df_keep = df[columns_to_keep]
        df_reduce = df.drop(columns_to_keep, axis=1)

        pca = PCA()
        pca.fit(df_reduce)
        n_components_to_keep = np.where(np.cumsum(pca.explained_variance_ratio_) > self.treshold)[0][0]

        self.pca = PCA(n_components=n_components_to_keep)
        print(f"Keeping {n_components_to_keep} components")

        df_reduce = pd.DataFrame(self.pca.fit_transform(df_reduce), columns=[f'PC{i}' for i in range(n_components_to_keep)])

        data = pd.concat([df_keep, df_reduce], axis=1)
        return data
    
    def predict(self, df):
        if self.pca is None:
            raise Exception("PCA not trained yet")
        
        columns_to_keep = [f"{self.to_forecast}_future{i}" for i in range(1, self.out_steps+1)] + ['Hour', 'Day Type']
        
        df_keep = df[columns_to_keep]
        df_reduce = df.drop(columns_to_keep, axis=1)

        df_reduce = pd.DataFrame(self.pca.transform(df_reduce), columns=[f'PC{i}' for i in range(self.pca.n_components_)])

        data = pd.concat([df_keep, df_reduce], axis=1)
        return data