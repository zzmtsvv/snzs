'''
Implementation of SincNet via arxiv.org/pdf/1808.00158
'''

%matplotlib inline
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import librosa
import IPython
import os
