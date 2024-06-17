import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Get the absolute path to the current directory (project root)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the dataset.csv file
dataset_path = os.path.join(current_dir, 'dataset_creation', 'phishy_website_dataset.csv')


data = pd.read_csv(dataset_path)

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

#initialising independent variables
x = data[req_cols].values

#initialising dependent variables
y = data['status'].values
#mapping legitimate as 0 and phishy as 1
mapping = {'Legitimate': 0, 'Phishy': 1}
y = np.array([mapping[label] for label in y])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)

#scaling down values to between -1 to 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)