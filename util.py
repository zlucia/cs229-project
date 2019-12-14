import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from featurize import get_feature_vector, parse_svg_path
from data import glyph_scraper
from torch.utils.data import Dataset
from torch import Tensor
import torchvision.transforms as transforms

# === Do not edit === #
DATASET_SIZE = 1883-22
TRAIN_SIZE = 1318-15
VAL_SIZE = 377-5
TEST_SIZE = 188-2

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
	def load(cls, embedding_path="data/vectors-200.tsv", image_path="data/font_images", knn_path="data/ensemble_dataset_norm.csv",
		metadata_path="data/metadata.tsv", glyph_path="data/font_glyphs", svg_data="data/svg_data.pkl"):
		print("Loading embeddings...", end="")
		fj_font_metadata = pd.read_csv(metadata_path, delimiter='\t', header=None, skiprows=1)
		cls.fj_font_names = fj_font_metadata.iloc[:, 0].to_frame()
		fj_font_vectors = pd.read_csv(embedding_path, delimiter='\t', header=None)
		cls.fj_font_data = pd.concat([fj_font_metadata, fj_font_vectors, ], axis=1, ignore_index=True).set_index([0])
		print("done")

		print("Loading typographic + semantic vectors...", end="")
		cls.knn_dataset = pd.read_csv(knn_path).set_index(['font_name'])
		cls.knn_dataset = cls.fj_font_names.merge(cls.knn_dataset, how="left", left_on=0, right_on="font_name").set_index([0])
		print("done")

		print("Loading images...", end="")
		if not os.path.isdir(image_path):
			print("Image data not found; ignoring...", end="")
		else:
			fj_font_image_filenames = pd.DataFrame([image_path + '/' + f for f in sorted(os.listdir(image_path))])
			cls.fj_images = pd.concat([cls.fj_font_names, fj_font_image_filenames, ], axis=1, ignore_index=True).set_index([0])
		print("done")

		print("Loading glyphs...", end="")
		if not os.path.isdir(glyph_path):
			print("Glyph data not found; ignoring...", end="")
		else:
			sorted_fj_font_names = pd.DataFrame([f for f in sorted(cls.fj_font_names.iloc[:, 0])])
			sorted_fj_font_glyph_filenames = pd.DataFrame([glyph_path + '/' + f + '/' for f in sorted(os.listdir(glyph_path),
				key=glyph_scraper.get_font_name_compare) if not f.startswith('.')])
			fj_glyphs = pd.concat([sorted_fj_font_names, sorted_fj_font_glyph_filenames, ], axis=1, ignore_index=True)
			cls.fj_glyphs = cls.fj_font_names.merge(fj_glyphs, how="inner", on=0).set_index([0])
		print("done")

		print("Loading SVGs...", end="")
		if not os.path.isfile(svg_data):
			print("SVG data not found; ignoring...", end="")
		else:
			fj_svgs = pd.read_pickle(svg_data)
			cls.fj_svgs = cls.fj_font_names.merge(fj_svgs, how="inner", on=0).set_index([0])
		print("done")

		invalid = [
			"Siemreap regular", "Content 700", "Bokor regular", "Suwannaphum regular", "Khmer regular",
			"Poller One regular", "Dangrek regular", "Moul regular", "Battambang regular", "Kdam Thmor regular",
			"Content regular", "Koulen regular", "Angkor regular", "Bayon regular", "Metal regular",
			"Andika regular", "Odor Mean Chey regular", "Chenla regular", "Battambang 700", "Taprom regular",
			"Freehand regular", "Moulpali regular"
		]

		cls.fj_font_names = cls.fj_font_names.loc[~cls.fj_font_names[0].isin(invalid)]
		remove_invalid = lambda x: cls.fj_font_names.merge(x, how="inner", on=0).set_index(0)
		
		cls.fj_font_data = remove_invalid(cls.fj_font_data)
		cls.knn_dataset = remove_invalid(cls.knn_dataset)
		if cls.fj_images is not None:
			cls.fj_images = remove_invalid(cls.fj_images)
		if cls.fj_glyphs is not None:
			cls.fj_glyphs = remove_invalid(cls.fj_glyphs)
		if cls.fj_svgs is not None:
			cls.fj_svgs = remove_invalid(cls.fj_svgs)

		def validate_data():
			eps = 1e-5
			assert len(cls.fj_font_names) == DATASET_SIZE
			assert cls.fj_font_names.values[0][0] == "Roboto 100"
			assert abs(cls.fj_font_data.values[0][2] - 3.4823321409e+2) <= eps
			# assert abs(cls.knn_dataset.values[0][1] - 0.01142927368) <= eps
			assert cls.fj_images is None or cls.fj_images.values[0][0].endswith("000000-font-0-100-Roboto.png")
			# assert cls.fj_glyphs is None or cls.fj_glyphs.values[0][0].endswith("Roboto-Thin/")
			# assert cls.fj_svgs is None or cls.fj_svgs.value1s[0][0]['A'].startswith("M967 435h")

		validate_data()

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
	def get_image_pil(cls, font_name):
		filename = cls.fj_images.loc[font_name].values[0]
		return Image.open(filename)

	@classmethod
	def get_glyph(cls, font_name, character):
		filename = os.path.join(cls.fj_glyphs.loc[font_name].values[0], "{}.png".format(ord(character)))
		return plt.imread(filename)[:, :, 0]

	@classmethod
	def get_glyph_pil(cls, font_name, character):
		filename = os.path.join(cls.fj_glyphs.loc[font_name].values[0], "{}.png".format(ord(character)))
		return Image.open(filename)

	@classmethod
	def get_svg_path(cls, font_name, character):
		return cls.fj_svgs.loc[font_name][1][character]

	@classmethod
	def get_svg(cls, font_name, character):
		return get_feature_vector(parse_svg_path(cls.get_svg_path(font_name, character)))

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
		return cls.knn_dataset.values[:, 0:6].take(cls.get_indices(kind), axis=0)

	@classmethod
	def get_all_image(cls, kind):
		return np.array([cls.get_image(font[0]) for font in cls.get_all_name(kind)])

	@classmethod
	def get_all_glyph(cls, kind, character):
		return np.array([cls.get_glyph(font[0], character) for font in cls.get_all_name(kind)])

	@classmethod
	def get_all_svg(cls, kind, character):
		return np.array([cls.get_svg(font[0], character) for font in cls.get_all_name(kind)])

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
		self.glyph_transformer = transforms.Compose([
    		transforms.Resize(64),
    		transforms.Grayscale(num_output_channels=1),
		    transforms.ToTensor()])
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
			image = self.data.get_image_pil(self.data.get_name(idx, self.kind))
			sample['image'] = self.glyph_transformer(image)
		if 'glyph' in self.types:
			glyph = self.data.get_glyph_pil(self.data.get_name(idx, self.kind), self.character)
			sample['glyph'] = self.glyph_transformer(glyph)
		if 'svg' in self.types:
			sample['svg'] = Tensor(self.data.get_svg(self.data.get_name(idx, self.kind), character=self.character))
		if 'semantic' in self.types:
			sample['semantic'] = self.semantic[idx]

		return sample
