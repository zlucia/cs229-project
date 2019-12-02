import numpy as np
import pandas as pd
import requests, io, zipfile
from collections import OrderedDict
from bs4 import BeautifulSoup
from importlib import reload
import util
reload(util)

data = util.FontData
FontDataset = util.FontDataset
data.load()
font_names_full = pd.DataFrame(np.sort(data.get_all_name('all'), axis=None).astype(str))
#print(font_names_full)
font_names_link = pd.DataFrame(font_names_full.iloc[:, 0].str.split().str[0:-1].str.join('-').str.lower())
font_names_link = font_names_link.drop_duplicates().reset_index(drop=True)
#print(font_names_link)

ff_url = 'https://www.1001freefonts.com/'
fs_url = 'https://www.fontsquirrel.com/fonts/download/'
urls = [ff_url, fs_url]
outfile_path = './font_files/'

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

def get_download_url(url, font_name):
	download_url = ''
	if url == ff_url:
		search_url = url + font_name + '.font'
		response = requests.get(search_url)
		soup = BeautifulSoup(response.text, 'html.parser')
		url_suffix = soup.select_one('a[href^="/d/"]')['href']
		download_url = url + url_suffix[1:]
	elif url == fs_url:
		download_url = url + font_name
	return download_url

def get_font_name_full(file):
	name_tokens = file[0:-4].split('-')
	for w, rw in mapping.items():
		if w in name_tokens[-1]:
			name_tokens[-1] = name_tokens[-1].replace(w, rw)
			if w == 'Condensed':
				name_tokens[0] += 'Condensed'
	name = ' '.join(name_tokens)
	return name

# Downloads OTF/TTF files
# for index in font_names_link.index:
#   font_name = font_names_link.iloc[index, 0]
font_name = 'abeezee'
for url in urls:
	download_url = get_download_url(url, font_name)

	try:
		r = requests.get(download_url)
		r.raise_for_status()
		z = zipfile.ZipFile(io.BytesIO(r.content))
		for file in z.namelist():
			name = get_font_name_full(file)
			# Extract font files for font names in font_names_full
			if name in font_names_full.values:
				z.extract(file, outfile_path)
		break
	except requests.exceptions.HTTPError as err:
		if url == urls[-1]:
			print('Font not found')

# .png files of all glyphs from .otf and .ttf
