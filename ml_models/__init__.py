from .base import *
from .regression import *
from .preprocessing import *

import sys
import inspect

preprocessor_lookup = None
models_lookup = None

def get_model_by_name(name):
	global models_lookup
	if not models_lookup:
		models_lookup = dict()
		for _, obj in inspect.getmembers(sys.modules['ml_models'], lambda x: inspect.isclass(x) and issubclass(x, BaseModel) and hasattr(x, 'model_name')):
			model_name = obj.model_name
			models_lookup[model_name] = obj
	return models_lookup[name]

def get_preprocessor_by_name(name):
	global preprocessor_lookup
	if not preprocessor_lookup:
		preprocessor_lookup = dict()
		for _, obj in inspect.getmembers(sys.modules['ml_models'], lambda x: inspect.isclass(x) and issubclass(x, BasePreprocessor) and hasattr(x, 'preprocessor_name')):
			preprocessor_name = obj.preprocessor_name
			preprocessor_lookup[preprocessor_name] = obj
	return preprocessor_lookup[name]