import sklearn.linear_model
import sklearn.ensemble

from .base import BaseRegression

class LinearRegression(BaseRegression):

    model_name = "linear-reg"

    def __init__(self, model_id, **kwargs):
        estimator = sklearn.linear_model.LinearRegression()
        super().__init__(model_id, estimator, **kwargs)    

class RandomForestRegression(BaseRegression):

    model_name = "random-forest-reg"

    def __init__(self, model_id, **kwargs):
        estimator = sklearn.ensemble.RandomForestRegressor()
        super().__init__(model_id, estimator, **kwargs)

class GradientBoostingRegression(BaseRegression):

    model_name = "gradient-boosting-reg"

    def __init__(self, model_id, **kwargs):
        estimator = sklearn.ensemble.GradientBoostingRegressor()
        super().__init__(model_id, estimator,**kwargs)

class AdaBoostRegression(BaseRegression):

    model_name = "ada-boost-reg"

    def __init__(self, model_id, **kwargs):
        estimator = sklearn.ensemble.AdaBoostRegressor()
        super().__init__(model_id, estimator, **kwargs)