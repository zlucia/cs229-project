import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataloader import get_dataloaders
from tqdm import tqdm
import train_util

from model import FontCNN, FontRNN, loss_fn
from experiment import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/sandbox',
	help="Directory that contains params.json")
parser.add_argument('--restore_file', default=None,
	help="File containing weights to reload")

def evaluate(model, loss_fn, dataloader, data_types, metrics, params):
	model.eval()

	overall_summary = []

	for i, batch in enumerate(dataloader):

		if data_types[0] == "glyph" or data_types[0] == "image":
			# Get batch
			X_batch = batch[data_types[0]]
			Y_batch = batch[data_types[1]]
			if params.use_gpu:
				X_batch = X_batch.cuda(non_blocking=True)
				Y_batch = Y_batch.cuda(non_blocking=True)
			X_batch = Variable(X_batch)
			Y_batch = Variable(Y_batch)

			# Predict
			Y_pred_batch = model(X_batch)

		elif data_types[0] == "svg":
			# Get batch
			X_batch = batch[data_types[0]]
			X_len = batch["len"]
			Y_batch = batch[data_types[1]]
			if params.use_gpu:
				X_batch = X_batch.cuda(non_blocking=True)
				X_len = X_len.cuda(non_blocking=True)
				Y_batch = Y_batch.cuda(non_blocking=True)
			X_batch = Variable(X_batch)
			Y_batch = Variable(Y_batch)

			# Predict
			Y_pred_batch = model(X_batch, X_len)

		loss = loss_fn(Y_pred_batch.float(), Y_batch.float())

		Y_pred_batch = Y_pred_batch.data.cpu()
		Y_batch = Y_batch.data.cpu()
		summary = {metric: metrics[metric](Y_pred_batch, Y_batch) for metric in metrics}
		summary['loss'] = loss.item()
		overall_summary.append(summary)

	metrics_mean = {metric: np.mean([s[metric] for s in overall_summary])
		for metric in overall_summary[0]}
	metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_mean.items())
	logging.info("- Eval metrics : " + metrics_string)

	return metrics_mean

if __name__ == "__main__":

	args = parser.parse_args()
	params_path = os.path.join(args.model_dir, 'params.json')
	params = train_util.Params(params_path)

	params.use_gpu = torch.cuda.is_available()

	train_util.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

	torch.manual_seed(0)
	if params.use_gpu:
		logging.info("GPU found")
		torch.cuda.manual_seed(0)
	else:
		logging.info("GPU not found")

	# At the end, manually change "val" to "test"
	test = "test"

	if params.data_type == "glyph_raster":
		data_types = ["image", "semantic"]
		dataloaders = get_dataloaders(params, [test], data_types, character=params.character)
	elif params.data_type == "glyph_vector":
		data_types = ["svg", "semantic"]
		dataloaders = get_dataloaders(params, [test], data_types, character=params.character)
	else:
		raise Exception("Invalid data type requested")

	test_dataloader = dataloaders[test]

	logging.info("- done")

	Model = None
	if params.model == "FontCNN":
		Model = FontCNN
	elif params.model == "FontRNN":
		Model = FontRNN

	model = Model(params).cuda() if params.use_gpu else Model(params)

	logging.info("Starting evaluation")

	train_util.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

	test_metrics = evaluate(model, loss_fn, test_dataloader, data_types, metrics, params)
	save_path = os.path.join(args.model_dir, "test_metrics_{}.json".format(args.restore_file))
	train_util.save_dict_to_json(test_metrics, save_path)
