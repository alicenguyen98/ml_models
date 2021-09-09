import abc
import re
import sklearn.base
import sklearn.model_selection
import sklearn.metrics
import numpy as np
import pandas as pd

#region model

class BaseModel(metaclass=abc.ABCMeta):
    """
    Skeleton of other models
    """
    def __init__(self, model_id: str, estimator: sklearn.base.BaseEstimator, **kwargs):
        self._model_id = model_id
        self._estimator = estimator
        self._trained = False

        # Set the remaining kwargs
        for k, v in kwargs.items():

            # Avoid override attribute
            if hasattr(self, k):
                raise Exception(f"Already has an attribute named {k}.")

            setattr(self, k, v)

    #region properties

    @property
    def model_id(self):
        """
        ID for this instance of model
        """
        return self._model_id
    
    @property
    def trained(self):
        """
        Is model trained?
        """
        return self._trained

    @property
    def estimator_params(self):
        """
        The params of the estimator. Can only be called after trained.
        """
        if not self._trained:
            raise Exception("Model not trained!")
        return self._estimator.get_params()
    
    @property
    def X_names(self):
        """
        The column names of the original X when first trained. Can only be called after trained.
        """
        if not self._trained:
            raise Exception("Model not trained!")
        return self._X_names
    
    @property
    def X_dtypes(self):
        """
        The column data types of the original X when first trained. Can only be called after trained.
        """
        if not self._trained:
            raise Exception("Model not trained!")
        return self._X_dtypes


    @property
    def feature_names(self):
        """
        The feature names (after pre-processed) when first trained. Can only be called after trained.
        """
        if not self._trained:
            raise Exception("Model not trained!")
        return self._feature_names

    @property
    def feature_dtypes(self):
        """
        The feature data types (after pre-processed) when first trained. Can only be called after trained.
        """
        if not self._trained:
            raise Exception("Model not trained!")
        return self._feature_dtypes

    #endregion

    #region static methods

    @staticmethod
    def _get_cv(mode, estimator, params, **kwargs):
        """
        Get cross validator by mode
        """
        if mode == 'random':
            return sklearn.model_selection.RandomizedSearchCV(estimator=estimator, param_distributions=params, **kwargs)
        elif mode == 'grid':
            return sklearn.model_selection.GridSearchCV(estimator=estimator, param_grid=params, **kwargs)
        else:
            raise NotImplementedError(f'Unknown hyperparameters tunning mode: {mode}')
    
    #endregion

    def train(self, X, y):
        """
        Train the model
        """
        # Save the inital column names and data types
        if isinstance(X, pd.DataFrame):
            self._X_names = X.columns
            self._X_dtypes = X.dtypes.to_dict()

        # Preprocess data
        if preprocessors := getattr(self, 'preprocessors', None):
            self._print(f'{self.model_id}: Trianing preprocessors')

            # Copy data to avoid corrupting original
            X = X.copy(deep=True)
            for preprocessor in preprocessors:
                preprocessor.train(X)
                X = preprocessor.transform(X)
            
            if getattr(self, 'verbose', False):
                X.info()

        # Save the processed feature names
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns
            self._feature_dtypes = X.dtypes.to_dict()

        # Train the model
        if (hpt := getattr(self, 'hpt', None)) and (mode := hpt.pop('mode', None)) and (params := getattr(self, 'params', None)):
            self._print(f'{self.model_id}: Hyperparameter tuning enabled (mode: {mode})')
            # Hyper-parameter tuning
            self._cv = BaseModel._get_cv(mode, self._estimator, self.params, **hpt)
            self._cv.fit(X=X, y=y)
            self._estimator = self._cv.best_estimator_
        else:
            # Static params
            if params := getattr(self, 'params', None):
                self._estimator.set_params(**self.params)
            self._estimator.fit(X, y)

        self._trained = True

    def predict(self, X):
        """
        Predict with trained models
        """            
        # Check if the model is already trained
        if not self._trained: 
            raise Exception("Model not trained!")

        X = self.preprocess(X)

        # Predict premium value
        regression = self._estimator.predict(X)

        return regression

    def preprocess(self, X):
        if not self._trained: 
            raise Exception("Model not trained!")

        # Preprocess X
        if preprocessors := getattr(self, 'preprocessors', None):
            # Copy data to avoid corrupting original
            X = X.copy(deep=True)
            for preprocessor in preprocessors:
                X = preprocessor.transform(X)
            
            if getattr(self, 'verbose', False):
                X.info()
            
        return X

    def get_performance(self, y_true, y_pred, scores: list) -> dict:
        raise NotImplementedError()

    def _print(self, message):
        if not getattr(self, 'verbose', False):
            return
        print(message)

class BaseRegression(BaseModel, metaclass=abc.ABCMeta):

    score_func = {
        'explained_variance': sklearn.metrics.explained_variance_score,
        'max_error': sklearn.metrics.max_error,
        'neg_mean_absolute_error': sklearn.metrics.mean_absolute_error,
        'neg_mean_squared_error': sklearn.metrics.mean_squared_error,
        'neg_root_mean_squared_error': sklearn.metrics.mean_squared_error,
        'neg_mean_squared_log_error': sklearn.metrics.mean_squared_log_error,
        'neg_median_absolute_error': sklearn.metrics.median_absolute_error,
        'r2': sklearn.metrics.r2_score,
        'neg_mean_poisson_deviance': sklearn.metrics.mean_poisson_deviance,
        'neg_mean_gamma_deviance': sklearn.metrics.mean_gamma_deviance,
        'neg_mean_absolute_percentage_error': sklearn.metrics.mean_absolute_percentage_error,
    }

    score_kwargs = {
        'neg_mean_squared_error': { 'squared': True },
        'neg_root_mean_squared_error': { 'squared': False }
    }

    @staticmethod
    def _get_score(score_name, y_true, y_pred):
        func = BaseRegression.score_func.get(score_name)
        kwargs = BaseRegression.score_kwargs.get(score_name)

        if kwargs:
            return func(y_true, y_pred, **kwargs)
        else:
            return func(y_true, y_pred)

    def get_performance(self, y_true, y_pred, scores: list) -> dict:
        return {s: BaseRegression._get_score(s, y_true, y_pred) for s in scores}

#endregion

#region pre-processor

class BasePreprocessor(metaclass=abc.ABCMeta):

    def __init__(self, preprocessor_id, preprocessor, **kwargs):
        self._preprocessor_id = preprocessor_id
        self._preprocessor = preprocessor
        self._trained = False

        # Set the remaining kwargs
        for k, v in kwargs.items():

            # Avoid override attribute
            if hasattr(self, k):
                raise Exception(f"Already has an attribute named {k}.")

            setattr(self, k, v)
    
    @property
    def preprocessor_id(self):
        return self._preprocessor_id

    @property
    def trained(self):
        return self._trained

    def train(self, X):
        if params := getattr(self, 'params', None):
            self._preprocessor.set_params(**params)

        columns = getattr(self, 'columns', X.columns)

        self._preprocessor.fit(X[columns])
        self._trained = True
    
    def transform(self, X):
        if not self._trained:
            raise Exception("Preprocessor not trained!")
        columns = getattr(self, 'columns', X.columns)
        X[columns] = self._preprocessor.transform(X[columns])
        return X.copy()
#endregion