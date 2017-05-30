import sys
import pandas

if __name__ == '__main__':

    file_predict_path = sys.argv[1] if len(sys.argv) > 1 else ''
    file_test_path = sys.argv[2] if len(sys.argv) > 2 else ''

    file_test_path = 'data/fasttext/data.test'
    file_predict_path = 'result/fasttext/data.test.predict'
    test_df = pandas.read_csv(file_test_path, header=None)
    predict_df = pandas.read_csv(file_predict_path, header=None)

    diff = 0
    for a, b in zip(list(test_df[0].map(lambda x: x.strip())), list(predict_df[0].map(lambda x: x.strip()))):
        if a != b:
            diff += 1

    print('Accuracy: {%s}' % format(100 - (diff / float(len(test_df[0])) * 100)))
