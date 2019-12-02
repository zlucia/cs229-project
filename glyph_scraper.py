import numpy as np
import pandas as pd
import requests, io, zipfile
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

# Downloads OTF/TTF files
# for index in font_names_link.index:
#   font_name = font_names_link.iloc[index, 0]
font_name = 'abhaya-libre'
for url in urls:
	download_url = get_download_url(url, font_name)

	try:
		r = requests.get(download_url)
		r.raise_for_status()
		outfile_path = './font_files/' + font_name
		z = zipfile.ZipFile(io.BytesIO(r.content))
		z.extractall(outfile_path)
		break
	except requests.exceptions.HTTPError as err:
		if url == urls[-1]:
			print('Font not found')

# Search through all downloaded files for fonts in font_names_full
# .png files of all glyphs from .otf and .ttf
