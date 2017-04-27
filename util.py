from HTMLParser import HTMLParser
import MySQLdb
import pandas
import numpy as np

DB = "iconic"
USER = "root"
PASSWD = ""

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
    data = data.replace('\n', ' ').replace('\r', '') if isinstance(data, basestring) else data
    data = strip_tags(data).encode('utf-8') if isinstance(data, basestring) else data
    return data


def create_train_csv(file_path, target=None):
    _create_csv(file_path, limit=10000, target=target)


def create_test_csv(file_path, target=None):
    _create_csv(file_path, limit=1000, offset=10000, target=target)


def _create_csv(file_path, **kwargs):
    if kwargs.get('target', '') == 'category':
        target_sql = '(SELECT t2.id_catalog_category FROM catalog_category t0 \
            LEFT JOIN catalog_category t2 ON t2.lft < t0.lft AND t2.rgt > t0.rgt \
            WHERE t0.id_catalog_category = p.primary_category AND t2.id_catalog_category != 1 ORDER BY t2.lft limit 1)'
    else:
        target_sql = 'p.fk_catalog_attribute_set'

    query = "SELECT %s category, b.name brand, p.name title, pd.description description \
            FROM catalog_product p \
            JOIN catalog_product_data pd ON pd.fk_catalog_product_set = p.fk_catalog_product_set \
            JOIN catalog_brand b ON p.fk_catalog_brand = b.id_catalog_brand \
            WHERE p.status = 'active' group by p.fk_catalog_product_set ORDER BY %s limit %d offset %s;" \
            % (target_sql, kwargs.get('order_by', 'RAND()'), kwargs.get('limit', 1000), kwargs.get('offset', 0))

    c = getDb()
    result_list_df = pandas.read_sql(query, c)
    result_list_df = result_list_df.dropna(how='any')
    result_list_df['text'] = result_list_df.apply(
        lambda r: normalize(r['brand']) + ' ' + normalize(r['title']) + ' ' + normalize(r['description']), axis=1)
    result_list_df.drop(['title', 'brand', 'description'], axis=1, inplace=True)
    result_list_df.to_csv(file_path, index=None, header=None, float_format='%d')
