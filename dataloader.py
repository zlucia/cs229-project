import torch
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

from util import FontData as data
from util import FontDataset


class PadCollate:
	def __call__(self, batch):
		sorted_batch = sorted(batch, key=lambda x: x["svg"].shape[0], reverse=True)
		sequences = [x["svg"] for x in sorted_batch]
		sequences_padded = pad_sequence(sequences, batch_first=True)
		sequence_lengths = torch.LongTensor([len(x) for x in sequences])
		labels = torch.Tensor(list(map(lambda x: x["semantic"], sorted_batch)))
		return {"svg": sequences_padded,
				"len": sequence_lengths,
				"semantic": labels}

def get_dataloaders(params, kinds, types, character=None):
	""" Load data and get train/val/test dataloaders """
	data.load()

	dataloaders = {}
	for kind in kinds:
		shuffle = kind == "train"
		if types[0] == "svg":
			dataloaders[kind] = DataLoader(FontDataset(data, kind, types, character=character),
				shuffle=shuffle, batch_size=params.batch_size, num_workers=params.num_workers,
				pin_memory=params.use_gpu, collate_fn=PadCollate(), drop_last=True)
		else:
			dataloaders[kind] = DataLoader(FontDataset(data, kind, types, character=character),
				shuffle=shuffle, batch_size=params.batch_size, num_workers=params.num_workers,
				pin_memory=params.use_gpu, drop_last=True)

	return dataloaders
