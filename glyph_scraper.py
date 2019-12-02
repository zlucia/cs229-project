import numpy as np
import pandas as pd
import requests, io, zipfile
import os
import string
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from collections import OrderedDict
from importlib import reload
import util
reload(util)

# Takes roughly 20 min to run
# Outputs .ttf font files to font_files directory, .png glyph files to font_glyphs directory

mapping = OrderedDict([
		('Thin', '100'),
		('ExtraLight', '200'),
		('Extralight', '200'),
		('Light', '300'),
		('Regular', 'regular'),
		('Medium', '500'),
		('SemiBold', '600'),
		('Semibold', '600'),
		('Bold', '700'),
		('ExtraBold', '800'),
		('Extrabold', '800'),
		('UltraBold', '800'),
		('Ultrabold', '800'),
		('Black', '900'),
		('OSF', ''),
		('Condensed', ''),
		('Italic', 'italic'),
		('It', 'italic'),
		('Oblique', 'italic'),
	])

def get_font_name_full(file):
	name_tokens = file[0:-4].split('-')
	for w, rw in mapping.items():
		if w in name_tokens[-1]:
			name_tokens[-1] = name_tokens[-1].replace(w, rw)
			if w == 'Condensed':
				name_tokens[0] += 'Condensed'
	name = ' '.join(name_tokens)
	return name

# Load font names
data = util.FontData
data.load()
font_names_full = pd.DataFrame(np.sort(data.get_all_name('all'), axis=None).astype(str))
print(font_names_full.shape)
font_names_link = pd.DataFrame(font_names_full.iloc[:, 0].str.split().str[0:-1].str.join('+'))
font_names_link = font_names_link.drop_duplicates().reset_index(drop=True)

# # Downloads font files (.otf/.ttf)
# gf_url = 'https://fonts.google.com/download?family='
# fontfiles_path = './font_files/'
# if not os.path.exists(fontfiles_path):
# 	os.makedirs(fontfiles_path)
# for index in font_names_link.index:
# 	font_name = font_names_link.iloc[index, 0]
# 	download_url = gf_url + font_name
# 	try:
# 		r = requests.get(download_url)
# 		r.raise_for_status()
# 		z = zipfile.ZipFile(io.BytesIO(r.content))
# 		for file in z.namelist():
# 			name = get_font_name_full(file)
# 			# Keep only font files whose font names are in dataset
# 			if name in font_names_full.values:
# 				z.extract(file, fontfiles_path)
# 	except requests.exceptions.HTTPError as err:
# 		print(err)

# # Extract per glyph .png files from font files
# point_size = 10
# fig_size = (128/600, 128/600)

# fontglyphs_path = './font_glyphs/'
# if not os.path.exists(fontglyphs_path):
# 	os.makedirs(fontglyphs_path)

# fontfiles_dir = os.fsencode(fontfiles_path)
# for file in os.listdir(fontfiles_dir):
# 	filename = os.fsdecode(file)
# 	if not filename.endswith('.ttf') and not filename.endswith('.otf'):
# 		continue

# 	# Using matplotlib
# 	font_path = fontglyphs_path + get_font_name_full(filename) + '/'
# 	if not os.path.exists(font_path):
# 		os.makedirs(font_path)

# 	fontfile_path = fontfiles_path + filename
# 	prop = font_manager.FontProperties(fname=fontfile_path)
# 	for char in string.ascii_letters:
# 		fig = plt.figure(figsize=fig_size, dpi=600)
# 		plt.figtext(0.5, 0.5, char, ha='center', va='center', fontproperties=prop, fontsize=point_size)
# 		image_path = font_path + str(ord(char)) + ".png"
# 		plt.savefig(image_path, dpi=600)
# 		plt.close()
