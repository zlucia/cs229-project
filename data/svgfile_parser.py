import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import string
import pandas as pd
import numpy as np
import glyph_scraper
import util
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count

def get_glyph_attributes(glyphs):
	glyph_attributes = {}
	for glyph in glyphs:
		glyph_attributes[glyph['glyph-name']] = glyph['d']
	return glyph_attributes

def parse_svg_file(file_path):
	print('Parsing ' + file_path)
	svg = open(file_path, 'r').read()
	soup = BeautifulSoup(svg, 'lxml')
	glyphs = soup.find_all('glyph', {'glyph-name': lambda x: x in list(string.ascii_letters)})
	attributes = get_glyph_attributes(glyphs)
	metadata = soup.find('font-face')
	if metadata:
		attributes['metadata'] = {attr:metadata[attr] for attr in metadata.attrs}
	return attributes

def parse_svg_files(df):
	return df.iloc[:, 0].apply(parse_svg_file)

def main():
	svg_path = 'data/font_svgs/'
	svg_data = 'data/svg_data.pkl'
	data = util.FontData
	data.load()

	sorted_fj_font_names = pd.DataFrame([f for f in sorted(data.get_all_name('all'))])
	sorted_fj_font_svg_filenames = pd.DataFrame([svg_path + f for f in sorted(os.listdir(svg_path), key=glyph_scraper.get_font_name_compare) if not f.startswith('.')])
	
	n_processes = cpu_count()
	df_split = np.array_split(sorted_fj_font_svg_filenames, n_processes)
	pool = Pool(n_processes)
	svg_attributes = pd.concat(pool.map(parse_svg_files, df_split))
	pool.close()
	pool.join()

	fj_svgs = pd.concat([sorted_fj_font_names, svg_attributes, ], axis=1, ignore_index=True).set_index([0])
	fj_svgs.to_pickle(svg_data)
	print('SVG data saved to ' + svg_data)

if __name__ == "__main__":
   main()