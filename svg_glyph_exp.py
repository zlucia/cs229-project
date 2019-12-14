import os
import string
import argparse
import sys
from subprocess import check_call, call

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default="test")

args = parser.parse_args()
exp = args.exp_name

for letter in string.ascii_letters[:3]:

	if letter in string.ascii_lowercase:
		model_dir = "experiments/{}/lower-{}".format(exp, letter)
	else:
		model_dir = "experiments/{}/upper-{}".format(exp, letter)

	os.makedirs(model_dir)

	params = """\'
	{{
		\"model\": \"FontRNN\",
		\"data_type\": \"glyph_vector\",
		\"label_size\": 31,
		\"character\": \"{}\",
		\"num_workers\": 2,
		\"batch_size\": 93,
		\"summary_steps\": 5,
		\"num_epochs\": 10,
		\"lr\": 0.001,
		\"lr_decay\": 0.9999,
		\"min_lr\": 0.00001,
		\"cnn_num_channels\": 64,
		\"rnn_hidden_size\": 256,
		\"rnn_num_layers\": 1,
		\"rnn_bidirectional\": true,
		\"dropout_rate\": 0.1
	}}\'""".format(letter)

	echo = "echo {} > {}/params.json".format(params, model_dir)
	call(echo, shell=True)

print("Created directory: experiments/{}/".format(exp))
check_call("ls experiments/{}".format(exp), shell=True)
print("Example param:")
check_call("cat experiments/{}/lower-a/params.json".format(exp), shell=True)

for letter in string.ascii_letters:

	if letter in string.ascii_lowercase:
		model_dir = "experiments/{}/lower-{}".format(exp, letter)
	else:
		model_dir = "experiments/{}/upper-{}".format(exp, letter)

	python = sys.executable
	train = "{} train.py --model_dir={}".format(python, model_dir)
	call(train, shell=True)