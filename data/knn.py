import csv
import numpy as np
import pandas as pd
import re
from collections import OrderedDict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn import metrics

##### DATA INGESTION
dir_path = 'data/'
fj_font_metadata = pd.read_csv(dir_path + 'metadata.tsv', delimiter='\t', header=None, skiprows=1)
fj_font_names = fj_font_metadata.iloc[:, 0].to_frame()
fj_font_vectors = pd.read_csv(dir_path + 'vectors-200.tsv', delimiter='\t', header=None)
fj_fonts = pd.concat([fj_font_names, fj_font_vectors], axis=1, ignore_index=True)

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
od_font_attributes = pd.read_csv(dir_path + 'estimatedAttributes.csv', delimiter=',', header=None, skiprows=1)
od_font_attributes.iloc[:, 0] = od_font_attributes.iloc[:, 0].apply(convert_name).to_frame()
od_font_names = od_font_attributes.iloc[:, 0].to_frame()

# Expect this to be 161 rows of common fonts b/w od and fj
shared_font_names = od_font_names.merge(fj_font_names, how='inner', on=0)
# Sanity check for debugging merge errors
# od_not_in_fj_font_names = od_font_names[~od_font_names.iloc[:, 0].isin(shared_font_names.iloc[:, 0])]
# print(od_not_in_fj_names)

fj_unlabeled_font_vectors = fj_fonts[~fj_font_names.iloc[:, 0].isin(shared_font_names.iloc[:, 0])].sort_values(by=0).reset_index(drop=True)
fj_labeled_font_vectors = fj_fonts[fj_font_names.iloc[:, 0].isin(shared_font_names.iloc[:, 0])].sort_values(by=0).reset_index(drop=True)
# Use the common font attribute labels from od with labeled font vector data in fj to find knns and predict attribute labels
common_attribute_labels = od_font_attributes.merge(shared_font_names, how='inner', on=0).sort_values(by=0).reset_index(drop=True)

##### KNN
def cosine_similarity(x, y):
	return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def get_weights_kulah(distances):
	k = distances.shape[1]
	numerator = (np.sum(distances, axis=1) - distances.T).T
	denominator = np.sum(distances)
	weights = (1 / (k - 1)) * (numerator/denominator)
	return weights

X = fj_labeled_font_vectors.iloc[:, 1:]
y = common_attribute_labels.iloc[:, 1:]
known_attribute_labels = od_font_attributes

# Dataset Store
ds_store = {}

# Chose k=4 since error appears to plateau there in od (Fig 6), chose cosine similarity to replicate od
knn_unweighted = KNeighborsRegressor(n_neighbors=4, weights='uniform', metric=cosine_similarity)
knn_unweighted.fit(X, y)
preds_1 = knn_unweighted.predict(fj_unlabeled_font_vectors.iloc[:, 1:])
predicted_attribute_labels = pd.DataFrame(np.concatenate((fj_unlabeled_font_vectors.iloc[:, 0:1], preds_1), axis=1))
ds_store['unweighted'] = pd.concat([predicted_attribute_labels, known_attribute_labels], axis=0)

knn_weighted_inv = KNeighborsRegressor(n_neighbors=4, weights='distance', metric=cosine_similarity)
knn_weighted_inv.fit(X, y)
preds_2 = knn_weighted_inv.predict(fj_unlabeled_font_vectors.iloc[:, 1:])
predicted_attribute_labels = pd.DataFrame(np.concatenate((fj_unlabeled_font_vectors.iloc[:, 0:1], preds_2), axis=1))
ds_store['weighted_inv'] = pd.concat([predicted_attribute_labels, known_attribute_labels], axis=0)

knn_weighted_kulah = KNeighborsRegressor(n_neighbors=4, weights=get_weights_kulah, metric=cosine_similarity)
knn_weighted_kulah.fit(X, y)
preds_3 = knn_weighted_kulah.predict(fj_unlabeled_font_vectors.iloc[:, 1:])
predicted_attribute_labels = pd.DataFrame(np.concatenate((fj_unlabeled_font_vectors.iloc[:, 0:1], preds_3), axis=1))
ds_store['weighted_kulah'] = pd.concat([predicted_attribute_labels, known_attribute_labels], axis=0)

##### DATA EXPORT
# Format column names
od_attribute_names = np.loadtxt(dir_path + 'attrNames.txt', dtype=str)
od_attribute_names = np.insert(od_attribute_names, 0, 'font_name')
typographic_features = np.asarray(['font_name', 'capitals', 'cursive', 'display', 'italic', 'monospace', 'serif'])
semantic_features = od_attribute_names[~np.isin(od_attribute_names, typographic_features)]
ordered_col_names =  np.hstack((typographic_features, semantic_features))

for ds, all_attribute_labels in ds_store.items():
	all_attribute_labels.columns = od_attribute_names
	# Columns re-ordered so that col 0 is font_name, cols [1:7] are typographic_features, cols [7:] are semantic_features
	knn_dataset = all_attribute_labels[ordered_col_names]
	# Normalize dataset
	knn_dataset = knn_dataset.set_index(['font_name'])
	knn_dataset = (knn_dataset-knn_dataset.mean())/knn_dataset.std()

	# # Export knn_dataset to csv file
	# # Note: randomize row ordering when splitting into train/test/validation, current row ordering has all predicted font data before known font data
	file_name = 'knn_dataset_' + ds + '.csv'
	knn_dataset.to_csv(dir_path + file_name, index=True)
