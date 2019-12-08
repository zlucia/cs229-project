import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from data import glyph_scraper
from torch.utils.data import Dataset

# === Do not edit === #
DATASET_SIZE = 1883
TRAIN_SIZE = 1318
VAL_SIZE = 377
TEST_SIZE = 188

np.random.seed(0)
ind = np.arange(DATASET_SIZE)
np.random.shuffle(ind)
train_ind = ind[:TRAIN_SIZE]
val_ind = ind[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
train_and_val_ind = ind[:TRAIN_SIZE+VAL_SIZE]
test_ind = ind[TRAIN_SIZE+VAL_SIZE:DATASET_SIZE]
# =================== #

class FontData:

	fj_font_names = None
	fj_font_data = None
	knn_dataset = None
	fj_images = None
	fj_glyphs = None
	fj_svgs = None

	@classmethod

	def load(cls, embedding_path="data/vectors-200.tsv", image_path="data/font_images/", knn_path="data/knn_dataset.csv", metadata_path="data/metadata.tsv", glyph_path="data/font_glyphs/", svg_data="data/svg_data.pkl"):
		print("Loading embeddings...", end="")
		if cls.fj_font_data is None:
			fj_font_metadata = pd.read_csv(metadata_path, delimiter='\t', header=None, skiprows=1)
			cls.fj_font_names = fj_font_metadata.iloc[:, 0].to_frame()
			fj_font_vectors = pd.read_csv(embedding_path, delimiter='\t', header=None)
			cls.fj_font_data = pd.concat([fj_font_metadata, fj_font_vectors, ], axis=1, ignore_index=True).set_index([0])
			cls.fj_font_data.sort_values(0)
		print("done")

		print("Loading typographic + semantic vectors...", end="")
		if cls.knn_dataset is None:
			cls.knn_dataset = pd.read_csv("data/knn_dataset.csv").set_index(['font_name'])
			cls.knn_dataset = cls.knn_dataset.merge(cls.fj_font_names, how="right", left_on="font_name", right_on=0).set_index("font_name").iloc[:, :-1] # better way to restrict the knn dataset?
			cls.knn_dataset.sort_values('font_name')
		print("done")

		print("Loading images...", end="")
		if cls.fj_images is None:
			if not os.path.isdir(image_path):
				print("Image data not found; ignoring...", end="")
			else:
				fj_font_image_filenames = pd.DataFrame([image_path + '/' + f for f in sorted(os.listdir(image_path))])
				cls.fj_images = pd.concat([cls.fj_font_names, fj_font_image_filenames, ], axis=1, ignore_index=True).set_index([0])
		print("done")

		print("Loading glyphs...", end="")
		if cls.fj_glyphs is None:
			if not os.path.isdir(glyph_path):
				print("Glyph data not found; ignoring...", end="")
			else:
				sorted_fj_font_names = pd.DataFrame([f for f in sorted(cls.fj_font_names.iloc[:, 0])])
				sorted_fj_font_glyph_filenames = pd.DataFrame([glyph_path + f + '/' for f in sorted(os.listdir(glyph_path), key=glyph_scraper.get_font_name_compare) if not f.startswith('.')])
				cls.fj_glyphs = pd.concat([sorted_fj_font_names, sorted_fj_font_glyph_filenames, ], axis=1, ignore_index=True).set_index([0])
		print("done")

		print("Loading SVGs...", end="")
		if cls.fj_svgs is None:
			if not os.path.isfile(svg_data):
				print("SVG data not found; ignoring...", end="")
			else:
				cls.fj_svgs = pd.read_pickle(svg_data)
		print("done")

	@classmethod
	def check_valid(cls, font_name):
		if font_name not in cls.fj_font_data.index:
			raise Exception(font_name, "not found")

	@classmethod
	def get_indices(cls, kind):
		if kind == "train":
			return train_ind
		elif kind == "train+val":
			return train_and_val_ind
		elif kind == "val":
			return val_ind
		elif kind == "test":
			return test_ind
		elif kind == "all":
			return ind

	@classmethod
	def get_name(cls, index, kind):
		i = cls.get_indices(kind)[index]
		return cls.fj_font_names.values[i][0]

	@classmethod
	def get_embedding(cls, font_name):
		cls.check_valid(font_name)
		return cls.fj_font_data.loc[font_name].values[2:]

	@classmethod
	def get_category(cls, font_name):
		cls.check_valid(font_name)
		return cls.fj_font_data.loc[font_name].values[1]

	@classmethod
	def get_semantic(cls, font_name):
		cls.check_valid(font_name)
		return cls.knn_dataset.loc[font_name].values[6:]

	@classmethod
	def get_typographic(cls, font_name):
		cls.check_valid(font_name)
		return cls.knn_dataset.loc[font_name].values[:5]

	@classmethod
	def get_image(cls, font_name):
		filename = cls.fj_images.loc[font_name].values[0]
		return plt.imread(filename)[:, :, 0]

	@classmethod
	def get_glyph(cls, font_name, character):
		filename = os.path.join(cls.fj_glyphs.loc[font_name].values[0], "{}.png".format(ord(character)))
		return plt.imread(filename)[:, :, 0]

	@classmethod
	def get_svg(cls, font_name, character):
		return None

	@classmethod
	def get_all_name(cls, kind):
		return cls.fj_font_names.values.take(cls.get_indices(kind), axis=0)

	@classmethod
	def get_all_embedding(cls, kind):
		return cls.fj_font_data.values[:, 2:].take(cls.get_indices(kind), axis=0)

	@classmethod
	def get_all_category(cls, kind):
		return cls.fj_font_data.values[:, 1].take(cls.get_indices(kind), axis=0)

	@classmethod
	def get_all_semantic(cls, kind):
		return cls.knn_dataset.values[:, 6:].take(cls.get_indices(kind), axis=0)

	@classmethod
	def get_all_typographic(cls, kind):
		return cls.knn_dataset.values[:, 0:5].take(cls.get_indices(kind), axis=0)

	@classmethod
	def get_all_image(cls, kind):
		return np.array([cls.get_image(font[0]) for font in cls.get_all_name(kind)])

	@classmethod
	def get_all_glyph(cls, kind, character):
		return np.array([cls.get_glyph(font[0], character) for font in cls.get_all_name(kind)])

	@classmethod
	def get_all_svg(cls, kind, character):
		return np.array([cls.get_glyph(font[0], character) for font in cls.get_all_name(kind)])

class FontDataset():

	def __init__(self, data, kind, types=['name', 'embedding', 'typographic', 'image', 'glyph', 'svg', 'semantic'], character='A'):
		# Load in all data at once for all kinds except images and svg
		self.data = data
		self.kind = kind
		self.types = types
		self.name = FontData.get_all_name(kind)
		self.embedding = FontData.get_all_embedding(kind)
		self.typographic = FontData.get_all_typographic(kind)
		self.semantic = FontData.get_all_semantic(kind)
		self.character = character
		assert len(self.embedding) == len(self.typographic) == len(self.semantic)

	def __len__(self):
		return len(self.embedding)

	def __getitem__(self, idx):
		sample = {}
		if 'name' in self.types:
			sample['name'] = self.name[idx][0]
		if 'embedding' in self.types:
			sample['embedding'] = self.embedding[idx]
		if 'typographic' in self.types:
			sample['typographic'] = self.typographic[idx]
		if 'image' in self.types:
			sample['image'] = self.data.get_image(self.data.get_name(idx, self.kind))
		if 'glyph' in self.types:
			sample['glyph'] = self.data.get_glyph(self.data.get_name(idx, self.kind), self.character)
		if 'svg' in self.types:
			sample['svg'] = self.data.get_svg(self.data.get_name(idx, self.kind), self.character)
		if 'semantic' in self.types:
			sample['semantic'] = self.semantic[idx]

		return sample