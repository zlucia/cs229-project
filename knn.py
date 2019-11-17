import csv
import numpy as np
import pandas as pd
import re
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors

##### DATA INGESTION
fj_font_metadata = pd.read_csv('metadata.tsv', delimiter='\t', header=None, skiprows=1)
fj_font_names = fj_font_metadata.iloc[:, 0].to_frame()
#print(fj_font_names.shape)
fj_font_vectors = pd.read_csv('vectors-200.tsv', delimiter='\t', header=None)
#print(fj_font_vectors.shape)
fj_fonts = pd.concat([fj_font_names, fj_font_vectors], axis=1, ignore_index=True)
#print(fj_fonts)

# od font name to fj font name conversion function
def convert_name(name):
	if '-' not in name or 'Caption' in name:
		name += '-regular'
	name_tokens = re.split('-|_', name)

	# convert token -1
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
	for w, rw in mapping.items():
		if w in name_tokens[-1]:
			name_tokens[-1] = name_tokens[-1].replace(w, rw)
			if w == 'Condensed':
				name_tokens[0] += 'Condensed'

	# convert token 0
	name_tokens[0] = ' '.join(re.findall(r'[A-Za-z]+|\d+\D*', name_tokens[0]))
	name_tokens[0] = re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', name_tokens[0])
	# deal with special cases
	if name_tokens[0] == "Bench Nine":
		name_tokens[0] = "BenchNine"
	if name_tokens[0] == "Tulpen":
		name_tokens[0] = "Tulpen One"
	
	name = ' '.join(name_tokens)
	return name

# od font data
od_font_attributes = pd.read_csv('estimatedAttributes.csv', delimiter=',', header=None, skiprows=1)
od_font_attributes.iloc[:, 0] = od_font_attributes.iloc[:, 0].apply(convert_name).to_frame()
od_font_names = od_font_attributes.iloc[:, 0].to_frame()

shared_font_names = od_font_names.merge(fj_font_names, how='inner', on=0)
# Expect this to be 161 rows of common fonts b/w od and fj
# print(shared_font_names.shape)
# Sanity check for debugging merge errors
# od_not_in_fj_font_names = od_font_names[~od_font_names.iloc[:, 0].isin(shared_font_names.iloc[:, 0])]
# print(od_not_in_fj_names)

fj_unlabeled_font_vectors = fj_fonts[~fj_font_names.iloc[:, 0].isin(shared_font_names.iloc[:, 0])].sort_values(by=0).reset_index(drop=True)
fj_labeled_font_vectors = fj_fonts[fj_font_names.iloc[:, 0].isin(shared_font_names.iloc[:, 0])].sort_values(by=0).reset_index(drop=True)
# Use the common font attribute labels from od with labeled font vector data in fj to find knns and predict attribute labels
od_attribute_labels = od_font_attributes.merge(shared_font_names, how='inner', on=0).sort_values(by=0).reset_index(drop=True)

##### KNN
X = fj_labeled_font_vectors.iloc[:, 1:]
# Chose k=4 since error appears to plateau there in od (Fig 6), chose cosine similarity to replicate od
# Could experiment with different values of k and other similarity metrics
knn_model = NearestNeighbors(n_neighbors=4, metric='cosine')
knn_model.fit(X)
# Indices of the knns for each unlabeled font
nns = knn_model.kneighbors(fj_unlabeled_font_vectors.iloc[:, 1:], return_distance=False)

# Takes the attribute labels of the knns for each unlabeled font, computes unweighted average over attribute labels
# Could experiment with weighted average, which was shown to produce lower error in od
np_od_attribute_labels = od_attribute_labels.iloc[:, 1:].to_numpy()
preds = np.take(np_od_attribute_labels, nns, axis=0).mean(axis=1)
predicted_attribute_labels = pd.DataFrame(np.concatenate((fj_unlabeled_font_vectors.iloc[:, 0:1], preds), axis=1))
known_attribute_labels = od_font_attributes
all_attribute_labels = pd.concat([predicted_attribute_labels, known_attribute_labels], axis=0)
# Expect this to be 1992 rows, since we've predicted for 1722 fonts with unknown attributes for fonts in fj not in od
# and know the attributes for 161 common fonts and an additional 39 fonts in od that are not in fj 
# (who od attribute labels weren't used in knn since their font vectors are not in fj)
# print(all_attribute_labels.shape)

# Format nicely
od_attribute_names = np.loadtxt('attrNames.txt', dtype=str)
od_attribute_names = np.insert(od_attribute_names, 0, 'font_name')
all_attribute_labels.columns = od_attribute_names
typographic_features = np.asarray(['font_name', 'capitals', 'cursive', 'display', 'italic', 'monospace', 'serif'])
semantic_features = od_attribute_names[~np.isin(od_attribute_names, typographic_features)]
ordered_col_names =  np.hstack((typographic_features, semantic_features))
# Columns re-ordered so that col 0 is font_name, cols [1:7] are typographic_features, cols [7:] are semantic_features
# Note: randomize row ordering when splitting into train/test/validation, current row ordering has all predicted font data before known font data
knn_dataset = all_attribute_labels[ordered_col_names]
knn_dataset = knn_dataset.set_index(['font_name'])
knn_dataset = (knn_dataset-knn_dataset.mean())/knn_dataset.std()

# Export knn_dataset to csv file
knn_dataset.to_csv('knn_dataset.csv', index=True)
