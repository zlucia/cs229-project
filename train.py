import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
import torch.nn.init as weight_init
from torch.autograd import Variable
from tqdm import tqdm
from time import sleep
import train_util
from dataloader import get_dataloaders

from model import FontCNN, FontRNN, loss_fn, lr_decay
from evaluate import evaluate
from experiment import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/sandbox',
	help="Directory that contains params.json")
parser.add_argument('--restore_file', default=None,
	help="File containing weights to reload")


def train(model, optimizer, loss_fn, dataloader, data_types, metrics, params, pause=False):
	""" Train for a single epoch"""
	model.train()

	epoch_summary = []
	loss_running_avg = train_util.RunningAverage()

	with tqdm(total=len(dataloader)) as t:

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

				# Forward prop
				Y_pred_batch = model(X_batch, pause)

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

				# Forward prop
				Y_pred_batch = model(X_batch, X_len)

			loss = loss_fn(Y_pred_batch.float(), Y_batch.float())
			optimizer.zero_grad()

			# Backward prop
			loss.backward()

			# Gradient step
			optimizer.step()

			# Evaluate summaries
			if i % params.summary_steps == 0:
				Y_pred_batch = Y_pred_batch.data.cpu()
				Y_batch = Y_batch.data.cpu()
				summary = {metric: metrics[metric](Y_pred_batch, Y_batch) for metric in metrics}
				summary['loss'] = loss.item()
				epoch_summary.append(summary)

			loss_running_avg.update(loss.item())
			t.set_postfix(loss='{:05.5f}'.format(loss_running_avg()))
			t.update()

	metrics_mean = {metric: np.mean([s[metric] for s in epoch_summary])
		for metric in epoch_summary[0]}
	metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_mean.items())
	logging.info("- Train metrics: " + metrics_string)


def train_eval(model, train_dataloader, val_dataloader, data_types, optimizer, loss_fn, metrics,
	params, model_dir, restore_file=None):
	""" Train on all epochs; evaluate on validation set after each epoch """
	if restore_file is not None:
		restore_path = os.path.join(args.model_dir, args.restore_file + ".pth.tar")
		logging.info("Restoring from {}".format(restore_path))
		train_util.load_checkpoint(restore_path, model, optimizer)

	best_val_RMSE = float("inf")

	for epoch in range(params.num_epochs):
		logging.info("Epoch {} / {}".format(epoch + 1, params.num_epochs))

		pause = False# epoch==1

		# Train for one epoch
		train(model, optimizer, loss_fn, train_dataloader, data_types, metrics, params, pause)
		
		# Evaluate for one epoch
		val_metrics = evaluate(model, loss_fn, val_dataloader, data_types, metrics, params)
		is_best = val_metrics['RMSE'] <= best_val_RMSE

		# Save weights
		train_util.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()}, is_best, model_dir)

		if is_best:
			logging.info("- New best RMSE")
			best_val_RMSE = val_metrics["RMSE"]
			best_metrics_path = os.path.join(model_dir, "best_val_metrics.json")
			train_util.save_dict_to_json(val_metrics, best_metrics_path)

		last_metrics_path = os.path.join(model_dir, "last_val_metrics.json")
		train_util.save_dict_to_json(val_metrics, last_metrics_path)

		optimizer = lr_decay(optimizer, params)


if __name__ == '__main__':

	args = parser.parse_args()
	params_path = os.path.join(args.model_dir, 'params.json')
	params = train_util.Params(params_path)

	params.use_gpu = torch.cuda.is_available()

	train_util.set_logger(os.path.join(args.model_dir, 'train.log'))

	torch.manual_seed(0)
	if params.use_gpu:
		logging.info("GPU found")
		torch.cuda.manual_seed(0)
	else:
		logging.info("GPU not found")

	logging.info("Loading data")

	if params.data_type == "glyph_raster":
		data_types = ["image", "semantic"]
		print ("Note : using images")
		dataloaders = get_dataloaders(params, ["train", "val"], data_types, character=params.character)
	elif params.data_type == "glyph_vector":
		data_types = ["svg", "semantic"]
		dataloaders = get_dataloaders(params, ["train", "val"], data_types, character=params.character)
	else:
		raise Exception("Invalid data type requested")

	train_dataloader = dataloaders["train"]
	val_dataloader = dataloaders["val"]

	logging.info("- done")

	Model = None
	if params.model == "FontCNN":
		Model = FontCNN
	elif params.model == "FontRNN":
		Model = FontRNN

	model = Model(params).cuda() if params.use_gpu else Model(params)


	optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0.05)

	logging.info("Starting training for {} epochs".format(params.num_epochs))
	train_eval(model, train_dataloader, val_dataloader, data_types, optimizer, loss_fn, metrics, params,
		args.model_dir, args.restore_file)