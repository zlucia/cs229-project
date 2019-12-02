import numpy as np
import pandas as pd
import requests, io, zipfile
import time
import os
import string
from PIL import Image, ImageFont
from collections import OrderedDict
from bs4 import BeautifulSoup
from importlib import reload
import util
reload(util)

data = util.FontData
data.load()
font_names_full = pd.DataFrame(np.sort(data.get_all_name('all'), axis=None).astype(str))
#print(font_names_full)
font_names_link = pd.DataFrame(font_names_full.iloc[:, 0].str.split().str[0:-1].str.join('+'))
font_names_link = font_names_link.drop_duplicates().reset_index(drop=True)
#print(font_names_link)

gwf_url = 'https://fonts.google.com/download?family='

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

# def get_download_url(url, font_name):
# 	download_url = ''
# 	if url == ff_url:
# 		search_url = url + font_name + '.font'
# 		response = requests.get(search_url)
# 		soup = BeautifulSoup(response.text, 'html.parser')
# 		url_suffix = soup.select_one('a[href^="/d/"]')['href']
# 		download_url = url + url_suffix[1:]
# 	elif url == fs_url:
# 		download_url = 
# 	return download_url

def get_font_name_full(file):
	name_tokens = file[0:-4].split('-')
	for w, rw in mapping.items():
		if w in name_tokens[-1]:
			name_tokens[-1] = name_tokens[-1].replace(w, rw)
			if w == 'Condensed':
				name_tokens[0] += 'Condensed'
	name = ' '.join(name_tokens)
	return name

# Downloads font files (.otf/.ttf)
fontfiles_path = './font_files/'
if not os.path.exists(fontfiles_path):
	os.makedirs(fontfiles_path)
# for index in font_names_link.index:
#   font_name = font_names_link.iloc[index, 0]
font_name = 'ABeeZee'
download_url = gwf_url + font_name
try:
	r = requests.get(download_url)
	r.raise_for_status()
	z = zipfile.ZipFile(io.BytesIO(r.content))
	for file in z.namelist():
		name = get_font_name_full(file)
		# Extract font files for font names in dataset
		if name in font_names_full.values:
			z.extract(file, fontfiles_path)
except requests.exceptions.HTTPError as err:
	print(err)
