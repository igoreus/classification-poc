from HTMLParser import HTMLParser
import MySQLdb
import pandas
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import *

DB = "iconic"
USER = "root"
PASSWD = ""

cachedStopWords = stopwords.words("english")

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def getDb():
    return MySQLdb.connect(user=USER, passwd=PASSWD, db=DB, charset='utf8')


def normalize(data):
    data = '' if data is None else data
    if isinstance(data, basestring):
        data = data.replace('\n', ' ').replace('\r', '')
        data = strip_tags(data)

        tokenizer = RegexpTokenizer(r'\w+')
        stemmers = (PorterStemmer(), SnowballStemmer("english"))

        data = ' '.join(tokenizer.tokenize(data))
        data = ' '.join([word for word in data.split() if word not in cachedStopWords])
        for stemmer in stemmers:
            data = ' '.join([stemmer.stem(word) for word in data.split()])

    return data.encode('utf-8')


def create_train_csv(file_path, **kwargs):
    kwargs['limit'] = kwargs.get('limit', 100000)
    kwargs['offset'] = kwargs.get('offset', 0)
    _create_csv(file_path, **kwargs)


def create_test_csv(file_path, **kwargs):
    kwargs['limit'] = kwargs.get('limit', 1000)
    kwargs['offset'] = kwargs.get('offset', 0)
    _create_csv(file_path, **kwargs)


def _create_csv(file_path, **kwargs):
    if kwargs.get('target', '') == 'first_level_category':
        target_sql = '(SELECT t2.id_catalog_category FROM catalog_category t0 \
            LEFT JOIN catalog_category t2 ON t2.lft < t0.lft AND t2.rgt > t0.rgt \
            WHERE t0.id_catalog_category = p.primary_category AND t2.id_catalog_category != 1 ORDER BY t2.lft limit 1)'
    elif kwargs.get('target', '') == 'category':
        target_sql = 'p.primary_category'
    else:
        target_sql = 'p.fk_catalog_attribute_set'

    query = "SELECT %s category, b.name brand, p.name title, pd.description  description \
            FROM catalog_product p \
            JOIN catalog_product_data pd ON pd.fk_catalog_product_set = p.fk_catalog_product_set \
            JOIN catalog_brand b ON p.fk_catalog_brand = b.id_catalog_brand \
            WHERE p.status = 'active' group by p.fk_catalog_product_set ORDER BY %s limit %d offset %s;" \
            % (target_sql, kwargs.get('order_by', 'RAND()'), kwargs.get('limit', 1000), kwargs.get('offset', 0))
    c = getDb()

    result_list_df = pandas.read_sql(query, c)
    result_list_df = result_list_df.dropna(how='any')
    result_list_df['description'] = result_list_df['description'].map(lambda x: normalize(x))
    result_list_df['brand'] = result_list_df['brand'].map(lambda x: normalize(x))
    result_list_df['title'] = result_list_df['title'].map(lambda x: normalize(x))
    result_list_df['text'] = result_list_df.apply(
        lambda r: r['brand'] + ' ' + r['title'] + ' ' + r['description'], axis=1)

    result_list_df.drop(['title', 'brand', 'description'], axis=1, inplace=True)
    result_list_df.to_csv(file_path, index=None, header=None, float_format='%d')
