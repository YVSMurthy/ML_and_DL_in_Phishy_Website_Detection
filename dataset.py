import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Get the absolute path to the current directory (project root)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the dataset.csv file
dataset_path1 = os.path.join(current_dir, 'dataset_creation', 'UCI_feature_extract_dataset.csv')
dataset_path2 = os.path.join(current_dir, 'dataset_creation', 'Kaggle_feature_extract_dataset.csv')
dataset_path3 = dataset_path2 = os.path.join(current_dir, 'dataset_creation', 'phishy_website_dataset.csv')


data = pd.read_csv(dataset_path1)
data_uk = pd.read_csv(dataset_path2)

#extracting out the required feature columns from the dataset
req_cols = [
    'length_url', 'length_hostname', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq',
    'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolumn',
    'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'https_token', 'ratio_digits_url', 'ratio_digits_host',
    'punycode', 'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains', 'prefix_suffix',
    'random_domain', 'shortening_service', 'path_extension','length_words_raw', 'shortest_words_raw', 'shortest_word_host',
    'shortest_word_path', 'longest_words_raw','longest_word_host', 'longest_word_path', 'avg_words_raw', 'avg_word_host',
    'avg_word_path', 'phish_hints','domain_in_brand', 'brand_in_subdomain', 'brand_in_path', 'suspecious_tld'
]

# #initialising independent variables
# x = data[req_cols].values
# x_uk = data_uk[req_cols].values

# #initialising dependent variables
# y = data['label'].values
# y_uk = data_uk['label'].values

#initialising independent variables
x = data.drop(columns=['url','label'])
x_uk = data_uk.drop(columns=['url','label'])

#initialising dependent variables
y = data['label'].values
y_uk = data_uk['label'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train_uk, x_test_uk, y_train_uk, y_test_uk = train_test_split(x_uk, y_uk, test_size=0.2, random_state=0)

#scaling down values to between -1 to 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train_uk = scaler.fit_transform(x_train_uk)
x_test_uk = scaler.transform(x_test_uk)