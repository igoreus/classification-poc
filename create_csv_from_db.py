import sys
import util

if __name__ == '__main__':
    file_path = sys.argv[1] if len(sys.argv) > 1 else ''
    file_test_path = sys.argv[2] if len(sys.argv) > 2 else ''
    print "Creating csv..."
    util.create_train_csv(file_path)
    util.create_test_csv(file_test_path)

