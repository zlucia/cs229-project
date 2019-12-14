
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

loss_fn = nn.MSELoss(reduction="sum")

def lr_decay(optimizer, params):
    for param_group in optimizer.param_groups:
        if param_group['lr']>params.min_lr:
            param_group['lr'] *= params.lr_decay
    return optimizer

class FontRNN(nn.Module):

	def __init__(self, params):
		super(FontRNN, self).__init__()
		self.hidden_size = params.rnn_hidden_size
		self.num_layers = params.rnn_num_layers
		self.bidirectional = True
		self.batch_size = params.batch_size
		self.label_size = params.label_size
		self.use_gpu = params.use_gpu
		self.lstm = nn.LSTM(7, self.hidden_size, batch_first=True, num_layers=self.num_layers,
			bidirectional=self.bidirectional)
		self.fc = nn.Linear(2*self.hidden_size, self.label_size)

	def init_hidden_cell(self):
		k = 2 if self.bidirectional else 1
		if self.use_gpu:
			hidden = Variable(torch.zeros(k, self.batch_size, self.hidden_size).cuda())
			cell = Variable(torch.zeros(k, self.batch_size, self.hidden_size).cuda())
		else:
			hidden = Variable(torch.zeros(k, self.batch_size, self.hidden_size))
			cell = Variable(torch.zeros(k, self.batch_size, self.hidden_size))
		return (hidden, cell)

	def forward(self, x, x_len):
		x_pack = pack_padded_sequence(x, x_len, batch_first=True)
		self.hidden = self.init_hidden_cell()
		lstm_out, self.hidden = self.lstm(x_pack, self.hidden)
		lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, padding_value=float('nan'))
		l_lstm_out = torch.stack([out[length-1,:] for out, length in zip(lstm_out, x_len)])
		y_pred = self.fc(l_lstm_out)
		return y_pred

class FontCNN(nn.Module):
	
	def __init__(self, params):
		super(FontCNN, self).__init__()
		self.num_channels = params.cnn_num_channels
		n=self.num_channels
		self.conv1 = nn.Conv2d(1, n, kernel_size=11, stride=2, padding=5)
		self.bn1 = nn.BatchNorm2d(n)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(n, 2*n, kernel_size=3, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(2*n)
		self.relu2 = nn.ReLU(inplace=True)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.fc5 = nn.Linear(2*n*8*8, 256)
		self.relu5 = nn.ReLU(inplace=True)
		self.dropout5 = nn.Dropout(p=params.dropout_rate)
		self.fc6 = nn.Linear(256, params.label_size)

		self.apply(self.init_func)
	
	def init_func(self, m, init_gain=0.02):
		if hasattr(m, 'weight'):
			torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
		elif hasattr(m, 'bias') and m.bias is not None:
			torch.nn.init.constant_(m.bias.data, 0.0)

	
	def forward(self, x, pause=False):
		if pause:
			import pdb; pdb.set_trace()
		
		x = self.relu1(self.bn1(self.conv1(x)))
		x = self.relu2(self.bn2(self.conv2(x)))
		x = self.maxpool2(x)
		x = x.view(32, -1)
		x = self.relu5(self.fc5(x))
		x = self.dropout5(x)
		x = self.fc6(x)

		return x