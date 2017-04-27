import sys
import util


# def normalize(list):
#     #
#     return map(
#         lambda x: x.replace('\n', ' ').replace('\r', '') if isinstance(x, basestring) else x,
#             map(lambda x: util.strip_tags(x).encode('utf-8') if isinstance(x, basestring) else x, list)
#     )

if __name__ == '__main__':

    file_path = sys.argv[1] if len(sys.argv) > 1 else ''
    print "Creating csv..."
    util._create_csv(file_path, order_by='id_catalog_product', limit='1000000')

