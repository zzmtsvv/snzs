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
import soundfile as sf
import shutil

pi = 3.141592653589793
e = 2.718281828459045

_TIMIT_PATH = 'data/lisa/data/timit/raw/TIMIT'


class Sinc_Conv(nn.Module):
  '''
  Sinc-сверткa (without bias)

  sr - частота дискретизации (по умолчанию 16000)
  out_channels - количество  sinc-фильтров
  kernel_size - длина sinc-фильтра
  in_channels - количество входных каналов (должен быть 1)
  '''

  @staticmethod
  def to_hz(mel):
    return 700 * (e ** (mel / 1127) - 1)
  
  @staticmethod
  def to_mel(hz):
    return 1127 * np.log(1 + hz / 700)
  
  def __init__(self, out_channels, kernel_size, in_channels=1,
               stride=1, padding=0, dilation=1, groups=1, 
               min_low_hz=50, min_band_hz=50, sr=16000):
    super(Sinc_Conv, self).__init__()

    self.kernel_size = kernel_size
    if not kernel_size % 2:
      self.kernel_size += 1 # симметричность фильтров
    
    self.stride = stride
    self.padding = padding
    self.out_channels = out_channels
    self.dilation = dilation
    self.sample_rate = sr
    self.min_low_hz = min_low_hz
    self.min_band_hz = min_band_hz

    # равномерно распределенная инициализация в мел-пространстве
    high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

    mel = np.linspace(self.to_mel(30), self.to_mel(high_hz),
                      self.out_channels + 1)
    hz = self.to_hz(mel)

    # окно Хэмминга
    n = torch.linspace(0, self.kernel_size / 2 - 1, steps=self.kernel_size // 2)
    self.window = 0.54 - 0.46 * torch.cos(2 * pi * n / self.kernel_size)

    # нижняя граница частоты и полос частот фильтра (out_channels, 1)
    self.low_hz = nn.Parameter(torch.tensor(hz[:-1]).view(-1, 1))
    self.band_hz = nn.Parameter(torch.tensor(np.diff(hz)).view(-1, 1))

    # половина оси времени
    n = 2 * pi * torch.arange(-(self.kernel_size - 1) / 2.0, 0).view(1, -1)
    self.n = n / self.sample_rate

  def forward(self, x):
    '''
    x - батч звуковой дорожки (batch_size, 1, n)
    '''
    self.n = self.n.to(x.device).float()
    self.window = self.window.to(x.device)

    low = (self.min_low_hz + torch.abs(self.low_hz)).float()
    high = (torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz),
                       self.min_low_hz, self.sample_rate / 2)).float()
    band = high - low
    band = band[:, 0]
    center = 2 * band.view(-1, 1)

    f1_low = torch.matmul(low.double(), self.n.double()).float()
    f2_high = torch.matmul(high, self.n)

    left = self.window * 2 * (torch.sin(f2_high) - torch.sin(f1_low)) / self.n
    right = torch.flip(left, dims=[1])

    band_pass = torch.cat([left, center, right], dim=1)
    band_pass = band_pass / (2 * band[:, None])

    filters = band_pass.view(self.out_channels, 1, self.kernel_size)

    out = F.conv1d(x, filters, stride=self.stride, padding=self.padding,
                   dilation=self.dilation, groups=self.groups)
    return out

  
class LayerNorm(nn.Module):
  '''
  Just BatchNorm, without momentum
  '''
  def __init__(self, features_shape, eps=1e-5):
    super(LayerNorm, self).__init__()
    self.gamma = nn.Parameter(torch.ones(features_shape))
    self.beta = nn.Parameter(torch.zeros(features_shape))
    self.eps = eps
  
  def forward(self, x):
    mean = x.mean((-1), keepdim=True)
    std = x.std((-1), keepdim=True)
    return self.gamma * (x - mean) / (std + self.eps) + self.beta

  
class ArseNet(nn.Module):
  '''
  section 4.2
  '''
  def __init__(self, input_dim=wlen):
    super(ArseNet, self).__init__()
    self.input_dim = input_dim
    current_dim = input_dim

    # convolution
    self.ln0 = LayerNorm(current_dim)
    self.sinc1 = Sinc_Conv(80, 251)
    current_dim = int((current_dim - 250) / 3)
    self.ln1 = LayerNorm([80, current_dim])
    self.conv2 = nn.Conv1d(80, 60, 5)
    current_dim = int((current_dim - 4) / 3)
    self.ln2 = LayerNorm([60, current_dim])
    self.conv3 = nn.Conv1d(60, 60, 5)
    current_dim = int((current_dim - 4) / 3)
    self.ln3 = LayerNorm([60, current_dim])

    # fully connected part
    self.fc1 = nn.Linear(current_dim, 2048)
    self.bn1 = nn.BatchNorm1d(2048, momentum=0.5)
    self.fc2 = nn.Linear(2048, 2048)
    self.bn2 = nn.BatchNorm1d(2048, momentum=0.5)
    self.fc3 = nn.Linear(2048, 462)


  def forward(self, x):
    batch = x.shape[0]
    seq_len = x.shape[1]

    x = self.ln0((x))
    x = x.view(batch, 1, seq_len)

    x = F.leaky_relu(self.ln1(F.max_pool1d(torch.abs(self.sinc1(x)), 3)))
    x = F.leaky_relu(self.ln2(F.max_pool1d(self.conv2(x), 3)))
    x = F.leaky_relu(self.ln3(F.max_pool1d(self.conv3(x), 3)))
    
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.bn1(x)
    x = F.leaky_relu(x)
    x = self.fc2(x)
    x = x.bn2(x)
    x = F.leaky_relu(x)
    x = self.fc3(x)
    x = F.softmax(x, dim=1)

    return x
 

def read_list(list_file):
  with open(list_file, 'r') as f:
    list_of_signals = [x.rstrip().upper() for x in f.readlines()]
  return list_of_signals


def ig_f(dir, files):
  return [x for x in files if os.path.isfile(os.path.join(dir, x))]


def copy_folder(in_folder, out_folder):
   if not os.path.isdir(out_folder):
     shutil.copytree(in_folder, out_folder, ignore=ig_f)


in_folder = _TIMIT_PATH
out_folder = '/content/ars'
list_file = '/content/TIMIT_all.scp'

list_of_signals = read_list(list_file)
copy_folder(in_folder, out_folder)

for i in range(len(list_of_signals)):

  wav_file = in_folder + '/' + list_of_signals[i]
  [signal, fs] = sf.read(wav_file)
  signal = signal.astype(np.float64)

  # normalization
  signal = signal / np.max(np.abs(signal))

  wrd_file = wav_file.replace('.WAV', '.WRD')
  wrd_sig = read_list(wrd_file)
  beg_sig = int(wrd_sig[0].split(' ')[0])
  end_sig = int(wrd_sig[-1].split(' ')[1])

  # remove silence
  signal = signal[beg_sig:end_sig]

  file_out = out_folder + '/' + list_of_signals[i]
  sf.write(file_out, signal, fs)

cw_len = 200
cw_shift = 10
fs = 16000
wlen = int(fs * cw_len / 1000)
wshift = int(fs * cw_shift / 1000)
lr =  0.001
alpha = 0.95
eps = 1e-7
epochs = 100
batch_size = 128
batches = 800
eval_epoch = 10

def create_batches(bs, folder, wave_list, snt_train, wlen, labels, factor):
  signal_batch = np.zeros([bs, wlen])
  labels_batch = np.zeros(bs)

  idx_arr = np.random.randint(snt_train, size=bs)
  rand_arr = np.random.uniform(1.0 - factor, 1.0 + factor, bs)

  for i in range(bs):
    [signal, fs] = sf.read(folder + wave_list[idx_arr[i]])

    snt_len = signal.shape[0]
    snt_beg = np.random.randint(snt_len- wlen - 1)
    snt_end = snt_beg + wlen

    # formatting to mono (not stereo)
    if len(signal.shape) > 1:
      signal = signal[:, 0]
    
    signal_batch[i, :] = signal[snt_beg:snt_end] * rand_arr[i]
    labels_batch[i] = labels[wave_list[idx_arr[i]].lower()]

  input = Variable(torch.from_numpy(signal_batch).float().cuda().contiguous())
  label = Variable(torch.from_numpy(labels_batch).float().cuda().contiguous())

  return input, label

wave_list_train = read_list('/content/TIMIT_train.scp')
snt_train = len(wave_list_train)

wave_list_test = read_list('/content/TIMIT_test.scp')
snt_test = len(wave_list_test)

cost = nn.NLLLoss()

labels = np.load('/content/TIMIT_labels.npy', allow_pickle=True).item()

optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=eps)

folder = out_folder + '/'


for epoch in range(epochs):
  
  flag_on_test = 0
  model.train()

  loss_sum, err_sum = 0, 0
  
  for i in range(batches):
    inp, label = create_batches(batch_size, folder, wave_list_train, snt_train,
                                wlen, labels, 0.2)
    out = model(inp)
    loss = cost(pred, label.long())

    pred = torch.max(out, dim=1)[1]
    err = torch.mean((pred != label.long()).float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_sum += loss.detach()
    err_sum += err.detach()

  total_loss = loss_sum / batches
  total_err = err_sum / batches

  if not epoch % eval_epoch:
    model.eval()

    flag_on_test = 1
    loss_sum, err_sum, err_snt = 0, 0, 0

    with torch.no_grad():
      for k in range(snt_test):
        signal, fs = sf.read(folder + wave_list_test[k])

        signal = torch.from_numpy(signal).float().cuda().contiguous()
        lab_batch = labels[wave_list_test[k]]
        
        start, stop = 0, wlen
        frames = int((signal.shape[0] - wlen) / wshift)

        arr = torch.zeros([batch_size, wlen]).float().cuda().contiguous()
        lab = Variable((torch.zeros(frames + 1)+ lab_batch).cuda().contiguous().long())
        out = Variable(torch.zeros(frames + 1, 462).float().cuda().contiguous())

        count_fr, total_count_fr = 0, 0
        while stop < signal.shape[0]:
          arr[count_fr, :] = signal[start:stop]
          start += wshift
          stop = start + wlen
          count_fr += 1
          total_count_fr += 1
          if count_fr == batch_size:
            inp = Variable(arr)
            out[tota_count_fr - batch_size: total_count_fr, :] = model(inp)
            count_fr = 0
            arr = torch.zeros([batch_size, wlen]).float().cuda().contiguous()
        
        if count_fr:
          inp = Variable(arr[0:count_fr])
          out[total_count_fr - count_fr:total_count_fr, :] = model(inp)
        
        pred = torch.max(out, dim=1)[1]
        loss = cost(out, label.long())
        err = torch.mean((pred != lab.long()).float())

        val, best_class = torch.max(torch.sum(out, dim=0), 0)
        err_snt = err_snt + (best_class != lab[0]).float()
        loss_sum += loss.detach()
        err_sum += err.detach()
      total_err_snt_dev = err_snt / snt_test
      total_loss_dev = loss_sum / snt_test
      total_err_dev = err_sum / snt_test
    a = f'epoch {epoch} loss_tr {total_loss} err_tr {total_err} loss_test {total_loss_dev} err_test {total_err_dev} err_test_snt {total_err_snt_dev}'
    print(a)
  else:
    print(f'epoch {epoch} loss_tr {total_loss} err_tr {total_err}')
