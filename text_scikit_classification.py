import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import util
import time
import pandas
from sklearn.externals import joblib

FILE_PATH = 'data/scikit/data.csv'
MODEL_PATH = 'data/scikit/sklearn_model.pkl'
FILE_PATH_TEST = 'data/scikit/data_test.csv'


def get_model(model='Perceptron'):
    if model == 'Perceptron':
        clf = Pipeline([
            ('vec', TfidfVectorizer(ngram_range=(1, 3), analyzer='char', use_idf=False)),
            ('clf', Perceptron()),
        ])
    elif model == 'SGDClassifier':
        clf = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
        ])
    elif model == 'MultinomialNB':
        clf = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])
    else:
        raise ValueError('Supported models: [Perceptron, SGDClassifier, MultinomialNB]')

    return clf


def load_dataset():
    if not os.path.isfile(FILE_PATH) or not os.path.isfile(FILE_PATH_TEST):
        util.create_train_csv(FILE_PATH, limit=50000)
        util.create_test_csv(FILE_PATH_TEST, limit=1000)

    test_df = pandas.read_csv(FILE_PATH, header=None)
    predict_df = pandas.read_csv(FILE_PATH_TEST, header=None)
    return list(test_df[1]), list(predict_df[1]), list(test_df[0]), list(predict_df[0])


if __name__ == '__main__':
    start = time.time()
    docs_train, docs_test, y_train, y_test = load_dataset()
    print("Load dataset: %.3f sec" % (time.time() - start))

    try:
        start = time.time()
        clf = joblib.load(MODEL_PATH)
        print("Load pre-trained model: %.3f sec" % (time.time() - start))
    except IOError:
        start = time.time()
        clf = get_model('SGDClassifier')
        clf.fit(docs_train, y_train)
        joblib.dump(clf, MODEL_PATH)
        print("Train model: %.3f sec" % (time.time() - start))

    start = time.time()
    y_predicted = clf.predict(docs_test)
    print("Predict time: %.3f sec" % (time.time() - start))

    print('Accuracy: %s' % metrics.accuracy_score(y_test, y_predicted))
