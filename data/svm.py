import os
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVR

def get_hyper_params(filename):
	hyper_params = {}
	if not os.path.exists(dir_path + filename):
		C_range = np.logspace(0, 4, 5)
		gamma_range = np.logspace(-10, -1, 10)
		param_grid = dict(gamma=gamma_range, C=C_range)
		cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
		grid = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid, cv=cv)
		for i in range(y.shape[1]):
			grid.fit(X, y.iloc[:, i])
			hyper_params[i] = grid.best_params_
			# print("The best parameters are %s with a score of %0.2f"
			#       % (grid.best_params_, grid.best_score_))
		with open(dir_path + filename, 'wb') as file:
			pickle.dump(hyper_params, file, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		hyper_params = None
		with open(dir_path + filename, 'rb') as file:
			hyper_params = pickle.load(file)
	return hyper_params

dir_path = 'data/'
fj_unlabeled_font_vectors = pd.read_pickle(dir_path + 'fj_ul_font_vectors.pkl')
fj_labeled_font_vectors = pd.read_pickle(dir_path + 'fj_l_font_vectors.pkl')
common_attribute_labels = pd.read_pickle(dir_path + 'common_attribute_labels.pkl')

X = fj_labeled_font_vectors.iloc[:, 1:]
y = common_attribute_labels.iloc[:, 1:]

hyper_params = get_hyper_params('hyper_params.pkl')

preds = pd.DataFrame(data=[])
preds = pd.concat([preds, fj_unlabeled_font_vectors.iloc[:, 0:1], ], axis=1)

for i in range(y.shape[1]):
	y_1d = y.iloc[:, i]
	model = SVR(kernel='rbf', C=hyper_params[i]['C'], gamma=hyper_params[i]['gamma'])
	model.fit(X, y_1d)
	attribute_preds = pd.DataFrame(model.predict(fj_unlabeled_font_vectors.iloc[:, 1:]))
	preds = pd.concat([preds, attribute_preds, ], axis=1, ignore_index=True)
all_attribute_labels = pd.concat([preds, common_attribute_labels], axis=0)

# Format column names
od_attribute_names = np.loadtxt(dir_path + 'attrNames.txt', dtype=str)
od_attribute_names = np.insert(od_attribute_names, 0, 'font_name')
typographic_features = np.asarray(['font_name', 'capitals', 'cursive', 'display', 'italic', 'monospace', 'serif'])
semantic_features = od_attribute_names[~np.isin(od_attribute_names, typographic_features)]
ordered_col_names =  np.hstack((typographic_features, semantic_features))

all_attribute_labels.columns = od_attribute_names
# Columns re-ordered so that col 0 is font_name, cols [1:7] are typographic_features, cols [7:] are semantic_features
all_attribute_labels = all_attribute_labels[ordered_col_names]
all_attribute_labels = all_attribute_labels.set_index(['font_name'])
file_name = 'svm_dataset.csv'
all_attribute_labels.to_csv(dir_path + file_name, index=True)