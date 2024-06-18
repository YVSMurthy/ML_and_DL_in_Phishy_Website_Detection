import numpy as np
import pandas as pd
import urllib
from bs4 import BeautifulSoup
import re
import os

def parse_url(url):
    parsed_url = urllib.parse.urlparse(url)

    # Length Parameters
    length_url = len(url)
    length_hostname = len(parsed_url.hostname)
    nb_dots = parsed_url.hostname.count('.')
    nb_hyphens = parsed_url.hostname.count('-')
    nb_at = parsed_url.hostname.count('@')
    nb_qm = parsed_url.hostname.count('?')
    nb_and = parsed_url.hostname.count('&')
    nb_or = parsed_url.hostname.count('|')
    nb_eq = parsed_url.hostname.count('=')
    nb_underscore = parsed_url.hostname.count('_')
    nb_tilde = parsed_url.hostname.count('~')
    nb_percent = parsed_url.hostname.count('%')
    nb_slash = parsed_url.path.count('/')
    nb_star = parsed_url.hostname.count('*')
    nb_colon = parsed_url.hostname.count(':')
    nb_comma = parsed_url.hostname.count(',')
    nb_semicolon = parsed_url.hostname.count(';')
    nb_dollar = parsed_url.hostname.count('$')
    nb_space = parsed_url.hostname.count(' ')
    nb_www = 1 if parsed_url.hostname.startswith('www.') else 0
    nb_com = 1 if parsed_url.hostname.endswith('.com') else 0
    nb_dslash = 1 if '//' in url else 0

    # HTTPS Token
    https_token = 1 if 'https' in url else 0

    # Ratio of Digits
    ratio_digits_url = sum(c.isdigit() for c in url) / len(url)
    ratio_digits_host = sum(c.isdigit() for c in parsed_url.hostname) / len(parsed_url.hostname)

    # Punycode
    punycode = 1 if parsed_url.hostname.encode('ascii') != parsed_url.hostname else 0

    # TLD in Path and Subdomain
    tld_in_path = 1 if parsed_url.path.endswith('.com') else 0
    tld_in_subdomain = 1 if parsed_url.hostname.endswith('.com') else 0

    # Abnormal Subdomain
    abnormal_subdomain = 1 if re.match(r'[^\w.-]', parsed_url.hostname) else 0

    # Number of Subdomains
    nb_subdomains = len(parsed_url.hostname.split('.'))

    # Prefix and Suffix
    prefix_suffix = 1 if parsed_url.hostname.startswith('www.') or parsed_url.hostname.endswith('.com') else 0

    # Random Domain
    random_domain = 1 if parsed_url.hostname.startswith('xn--') else 0

    # Shortening Service
    shortening_service = 1 if parsed_url.netloc in ['bit.ly', 'goo.gl', 't.co'] else 0

    # Path Extension
    path_extension = 1 if parsed_url.path.endswith(('.html', '.php', '.asp', '.aspx', '.jsp')) else 0
    
    # Length of Words
    words = re.findall(r'\w+', parsed_url.geturl())
    length_words_raw = sum(len(word) for word in words)
    shortest_words_raw = min(len(word) for word in words)
    shortest_word_host = min(len(word) for word in parsed_url.hostname.split('.'))
    shortest_word_path = min(len(word) for word in parsed_url.path.split('/'))
    longest_words_raw = max(len(word) for word in words)
    longest_word_host = max(len(word) for word in parsed_url.hostname.split('.'))
    longest_word_path = max(len(word) for word in parsed_url.path.split('/'))
    avg_words_raw = length_words_raw / len(words) if len(words) > 0 else 0
    avg_word_host = length_hostname / parsed_url.hostname.count('.') if parsed_url.hostname.count('.') > 0 else 0
    avg_word_path = len(parsed_url.path) / parsed_url.path.count('/') if parsed_url.path.count('/') > 0 else 0

    # Phishing Hints
    phish_hints = 1 if 'paypal' in url or 'login' in url or 'confirm' in url else 0

    # Domain in Brand
    domain_in_brand = 1 if 'example' in parsed_url.hostname else 0

    # Brand in Subdomain and Path
    brand_in_subdomain = 1 if 'example' in parsed_url.hostname else 0
    brand_in_path = 1 if 'example' in parsed_url.path else 0

    # Suspicious TLD
    suspicious_tld = 1 if parsed_url.netloc.endswith(('info', 'xyz', 'online')) else 0

    # Statistical Report
    statistical_report = {
        'length_url': length_url,
        'length_hostname': length_hostname,
        'nb_dots': nb_dots,
        'nb_hyphens': nb_hyphens,
        'nb_at': nb_at,
        'nb_qm': nb_qm,
        'nb_and': nb_and,
        'nb_or': nb_or,
        'nb_eq': nb_eq,
        'nb_underscore': nb_underscore,
        'nb_tilde': nb_tilde,
        'nb_percent': nb_percent,
        'nb_slash': nb_slash,
        'nb_star': nb_star,
        'nb_colon': nb_colon,
        'nb_comma': nb_comma,
        'nb_semicolumn': nb_semicolon,
        'nb_dollar': nb_dollar,
        'nb_space': nb_space,
        'nb_www': nb_www,
        'nb_com': nb_com,
        'nb_dslash': nb_dslash,
        'https_token': https_token,
        'ratio_digits_url': ratio_digits_url,
        'ratio_digits_host': ratio_digits_host,
        'punycode': punycode,
        'tld_in_path': tld_in_path,
        'tld_in_subdomain': tld_in_subdomain,
        'abnormal_subdomain': abnormal_subdomain,
        'nb_subdomains': nb_subdomains,
        'prefix_suffix': prefix_suffix,
        'random_domain': random_domain,
        'shortening_service': shortening_service,
        'path_extension': path_extension,
        # 'ssl': ssl,
        # 'nb_redirection': nb_redirection,
        # 'nb_external_redirection': nb_external_redirection,
        'length_words_raw': length_words_raw,
        'shortest_words_raw': shortest_words_raw,
        'shortest_word_host': shortest_word_host,
        'shortest_word_path': shortest_word_path,
        'longest_words_raw': longest_words_raw,
        'longest_word_host': longest_word_host,
        'longest_word_path': longest_word_path,
        'avg_words_raw': avg_words_raw,
        'avg_word_host': avg_word_host,
        'avg_word_path': avg_word_path,
        'phish_hints': phish_hints,
        'domain_in_brand': domain_in_brand,
        'brand_in_subdomain': brand_in_subdomain,
        'brand_in_path': brand_in_path,
        'suspecious_tld': suspicious_tld
    }

    return statistical_report

current_dir = os.path.dirname(os.path.abspath(__file__))
original_file = os.path.join(current_dir, 'website_list3.csv')
data = pd.read_csv(original_file)
# data = pd.read_excel(original_file)

mapping = {'Legitimate': 1, 'Phishy': 0}
data['label'] = data['label'].map(mapping)
# print(data)

#shuffling the data
data = data.sample(frac=1).reset_index(drop=True)

output = []
for i in range(len(data)):
    url = data['URL'][i]
    label = data['label'][i]
    print(i, " ", url)
    url_details = [url] + list(parse_url(url).values()) + [label]
    output.append(url_details)

req_cols = [
    'url', 'length_url', 'length_hostname', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq','nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolumn','nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'https_token', 'ratio_digits_url', 'ratio_digits_host','punycode', 'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains', 'prefix_suffix', 'random_domain', 'shortening_service', 'path_extension','length_words_raw', 'shortest_words_raw', 'shortest_word_host','shortest_word_path', 'longest_words_raw','longest_word_host', 'longest_word_path', 'avg_words_raw', 'avg_word_host', 'avg_word_path', 'phish_hints', 'domain_in_brand', 'brand_in_subdomain', 'brand_in_path', 'suspecious_tld', 'label'
]

output = pd.DataFrame(output, columns=req_cols)
# print(output)
output_path = os.path.join(current_dir, 'Kaggle_feature_extract_dataset.csv')
output.to_csv(output_path, index=False)