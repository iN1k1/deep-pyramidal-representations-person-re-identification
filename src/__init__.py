from .datamanager import *
from .ml import *
from .results import *
from .visualization import *
import pickle

def save(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj.__dict__, f)

def load(obj, file_path):
    with open(file_path, 'rb') as f:
        tmp_dict = pickle.load(f)
        obj.__dict__.update(tmp_dict)