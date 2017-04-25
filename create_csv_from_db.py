import sys
import util
import pandas

# def normalize(list):
#     #
#     return map(
#         lambda x: x.replace('\n', ' ').replace('\r', '') if isinstance(x, basestring) else x,
#             map(lambda x: util.strip_tags(x).encode('utf-8') if isinstance(x, basestring) else x, list)
#     )

def normalize(data):
    data = '' if data is None else data
    data = data.replace('\n', ' ').replace('\r', '') if isinstance(data, basestring) else data
    data = util.strip_tags(data).encode('utf-8') if isinstance(data, basestring) else data
    return data



if __name__ == '__main__':

    file_path = sys.argv[1] if len(sys.argv) > 1 else ''
    print "Getting csv..."
    query = """SELECT p.primary_category category, b.name brand, p.name title, pd.description description
            FROM catalog_product p
            JOIN catalog_product_data pd ON pd.fk_catalog_product_set = p.fk_catalog_product_set
            JOIN catalog_brand b ON p.fk_catalog_brand = b.id_catalog_brand
            WHERE p.status = 'active' group by p.fk_catalog_product_set ORDER BY id_catalog_product;"""

    c = util.getDb()
    result_list_df = pandas.read_sql(query, c)

    result_list_df['text'] = result_list_df.apply(
        lambda r: normalize(r['brand']) + ' ' + normalize(r['title']) + ' ' + normalize(r['description']), axis=1)
    result_list_df.drop(['title', 'brand', 'description'], axis=1, inplace=True)
    result_list_df.to_csv(file_path, index=None, header=None)
