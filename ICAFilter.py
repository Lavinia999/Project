
import pandas as pd
from sklearn.decomposition import FastICA

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class ICAFilter():
    def __init__(self, n_components : int = None):
        '''
            Setting the algorithm
        :param n_components: integer, by default = None
            Number of components to keep
        '''
        self.n_components = n_components

    def filter(self, dataframe : pd.DataFrame, y_column : str):
        '''
            Creating filter to new data and separating superimposed signals
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The column name of the value that we have to predict
        '''
        #Splitting dataframe
        self.dataframe = dataframe.copy()
        self.y_column = y_column
        self.X_columns = [col for col in self.dataframe.columns if col != self.y_column]
        self.X = self.dataframe[self.X_columns].values
        self.y = self.dataframe[y_column].values
        #Creating filter
        self.ica = FastICA(n_components = self.n_components)
        self.ica.fit(self.X)
        #Creating new data based on the filter
        X_ica = self.ica.transform(self.X)
        X_new = self.ica.inverse_transform(X_ica)
        X_new = pd.DataFrame(X_new, columns=self.X_columns)
        #Create and return new Dataframe
        X_new[y_column] = self.y
        return X_new

    def apply(self, dataframe : pd.DataFrame):
        '''
            Separating superimposed signals
            based on an already existed filter
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The column name of the value that we have to predict
        '''
        #Splitting data
        self.dataframe = dataframe.copy()
        X_columns = [col for col in self.dataframe.columns if col != self.y_column]
        X = self.dataframe[X_columns].values
        y = self.dataframe[self.y_column].values
        #Applying filter to the new dataframe
        X_ica = self.ica.transform(X)
        X_new = self.ica.inverse_transform(X_ica)
        X_new = pd.DataFrame(X_new, columns=self.X_columns)
        #Create and return new Dataframe
        X_new[self.y_column] = y
        return X_new