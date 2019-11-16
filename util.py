import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class FontData:

	fj_fonts = None
	knn_dataset = None
	fj_images = None

	@classmethod
	def load(cls, embedding_path="vectors-200.tsv", image_path="images/", knn_path="knn_dataset.csv", metadata_path="metadata.tsv"):
		print("Loading embeddings...", end="")
		if cls.fj_fonts is None:
			fj_font_metadata = pd.read_csv(metadata_path, delimiter='\t', header=None, skiprows=1)
			fj_font_names = fj_font_metadata.iloc[:, 0].to_frame()
			fj_font_vectors = pd.read_csv(embedding_path, delimiter='\t', header=None)
			cls.fj_fonts = pd.concat([fj_font_metadata, fj_font_vectors, ], axis=1, ignore_index=True).set_index([0])
			cls.fj_fonts.sort_values(0)
		print("done")

		print("Loading geometric + semantic vectors...", end="")
		if cls.knn_dataset is None:
			cls.knn_dataset = pd.read_csv("knn_dataset.csv").set_index(['font_name'])
			cls.knn_dataset.sort_values('font_name')
		print("done")

		print("Loading images...", end="")
		if cls.fj_images is None:
			fj_font_image_filenames = pd.DataFrame([image_path + '/' + f for f in sorted(os.listdir(image_path))])
			cls.fj_images = pd.concat([fj_font_names, fj_font_image_filenames, ], axis=1, ignore_index=True).set_index([0])
		print("done")

	@classmethod
	def check_valid(cls, font_name):
		if font_name not in cls.fj_fonts.index:
			raise Exception(font_name, "not found")

	@classmethod
	def get_embedding(cls, font_name):
		cls.check_valid(font_name)
		return cls.fj_fonts.loc[font_name].values[2:]

	@classmethod
	def get_category(cls, font_name):
		cls.check_valid(font_name)
		return cls.fj_fonts.loc[font_name].values[1]

	@classmethod
	def get_semantic(cls, font_name):
		cls.check_valid(font_name)
		return cls.knn_dataset.loc[font_name].values[6:]

	@classmethod
	def get_geometric(cls, font_name):
		cls.check_valid(font_name)
		return cls.knn_dataset.loc[font_name].values[:5]

	@classmethod
	def get_image(cls, font_name):
		filename = cls.fj_images.loc[font_name].values[0]
		return plt.imread(filename)

	@classmethod
	def get_all_embedding(cls):
		return cls.fj_fonts.values[:, 2:]

	@classmethod
	def get_all_category(cls):
		return cls.fj_fonts.values[:, 1]

	@classmethod
	def get_all_semantic(cls):
		return cls.knn_dataset.values[:, 6:]

	@classmethod
	def get_all_geometric(cls):
		return cls.knn_dataset.values[:, 0:5]

	@classmethod
	def get_all_image(cls):
		return np.array([cls.get_image(font) for font in cls.fj_images.index])

	@classmethod
	def get_font_names(cls):
		return cls.fj_fonts.index.values

