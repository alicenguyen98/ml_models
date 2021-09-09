import sklearn.preprocessing
import pandas as pd
import numpy as np

from .base import BasePreprocessor

class OneHotEndcoder(BasePreprocessor):

    preprocessor_name = 'one-hot-encoding'

    def __init__(self, preprocessor_id, **kwargs):
        preprocessor = sklearn.preprocessing.OneHotEncoder()
        super().__init__(preprocessor_id, preprocessor, **kwargs)

    def transform(self, X):
        if not self._trained:
            raise Exception("Preprocessor not trained!")
        old_columns = getattr(self, 'columns', X.columns)
        new_columns = self._preprocessor.get_feature_names(old_columns)
        X[new_columns] = self._preprocessor.transform(X[old_columns])
        X = X.drop(columns=old_columns)
        return X.copy()

class StandardScalingPreprocessor(BasePreprocessor):

    preprocessor_name = 'standard-scaling'

    def __init__(self, preprocessor_id, **kwargs):
        preprocessor = sklearn.preprocessing.StandardScaler()
        super().__init__(preprocessor_id, preprocessor, **kwargs)

class PolynomialFeaturePreprocessor(BasePreprocessor):
    
    preprocessor_name = 'polynomial'

    def __init__(self, preprocessor_id, **kwargs):
        super().__init__(preprocessor_id, None, **kwargs)

    def train(self, X):
        # Doesn't need training
        pass

    def transform(self, X):

        params = getattr(self, 'params', dict())
        degree = params.get('degree', 2)

        # convert X to DataFrame if it isn't
        if isinstance(X, (list, np.ndarray, pd.Series)):
            X = pd.DataFrame(X)

        columns = getattr(self, 'columns', X.columns)
        intersection = params.get('intersection', False)
        one = params.get('one', False)

        deg = 2
        while deg < degree + 1:
            # handle each degree
            for col in columns:
                dtype = PolynomialFeaturePreprocessor._get_64_dtype(X[col])
                col_name = f'{col}^{deg}'
                X[col_name] = X[col].astype(dtype).apply(lambda x: x ** deg)
                if dtype == np.int64:
                    X[col_name] = X[col_name].apply(pd.to_numeric, downcast='integer')
            deg += 1
            
        # handle intersection
        if intersection:
            for a in range(len(columns) - 1):
                b = a + 1
                while b < len(columns):
                    deg_a = 1
                    deg_b = degree - 1
                    while deg_a < degree:
                        dtype_a = PolynomialFeaturePreprocessor._get_64_dtype(X[columns[a]])
                        dtype_b = PolynomialFeaturePreprocessor._get_64_dtype(X[columns[b]])
                        col_name = f'{columns[a]}^{deg_a}{columns[b]}^{deg_b}'
                        X[col_name] = (X[columns[a]].astype(dtype_a) ** deg_a) * (X[columns[b]].astype(dtype_b) ** deg_b)

                        if dtype_a == np.int64 and dtype_b == np.int64:
                            X[col_name] = X[col_name].apply(pd.to_numeric, downcast='integer')
                        deg_a += 1
                        deg_b -= 1
                    b += 1

        if one:
            X['1'] = 1

        return X.copy()
    
    @staticmethod
    def _get_64_dtype(col):
        """
        Convert dtype to avoid integer overflow or precision problem
        """
        dtype = col.dtype
        if dtype in [np.int8, np.int16, np.int32]:                             
            return np.int64
        elif dtype in [np.float16, np.float32]:
            return np.float64
        return dtype