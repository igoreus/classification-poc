import fasttext
import util
from flask import Flask, request, jsonify
import logging

HOST = '127.0.0.1'
PORT = 3000
MODEL = 'result/fasttext/data.bin'

app = Flask(__name__)
logging.basicConfig(filename='/tmp/flask.log', format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y/%m/%d %I:%M:%S', level=logging.INFO)
predict_model = fasttext.load_model(MODEL)

@app.route("/")
def hello():
    logging.info('Hello World!')
    return "Hello World!"

@app.route('/health')
def health():
    return jsonify({'health': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    post_data = request.get_json(force=True)
    return jsonify({'status': 'ok', 'results': fast_predict(post_data)})


def fast_predict(data):
    query = """SELECT t0.name , GROUP_CONCAT(t2.id_catalog_category ORDER BY t2.lft) ancestor_ids, GROUP_CONCAT(t2.name ORDER BY t2.lft) ancestors
FROM catalog_category t0
LEFT JOIN catalog_category t2 ON t2.lft < t0.lft AND t2.rgt > t0.rgt
WHERE t0.id_catalog_category = %s"""

    db = util.getDb()
    c = db.cursor()

    item_texts = []

    for attrs in data:
        item_texts.append(attrs.get('brand') + ' ' + attrs.get('title') + ' ' +
                          util.strip_tags(attrs.get('description', '')).encode('utf-8'))
    labels = [v[0] for v in predict_model.predict(item_texts)]

    results = []
    for label in labels:
        category_id = label.replace('__label__', '')
        c.execute(query % category_id)
        category = c.fetchone(),
        results.append(
            {
                'category_id': category_id,
                'category_name': category[0][0],
                'category_ancestor_ids': category[0][1],
                'category_ancestors': category[0][2],
             }
        )

    return results

if __name__ == '__main__':
    logging.info('Starting a server...')
    app.run(host=HOST, port=PORT)


