import argparse
import os
from wrapper import MultiClass


parser = argparse.ArgumentParser(description='Train multiclass classifier')
parser.add_argument('-train', required=True, metavar='PATH', help='path to input train file')
parser.add_argument('-test', required=False, metavar='PATH', help='path to input test file')

args = parser.parse_args()

def load_data(file_path):
    X, Y = [], []
    with open(file_path, 'r') as f:
        for record in f:
            X.append(record.split('\t')[0])
            Y.append(record.split('\t')[1])
    print(X)
    return X, Y

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    X_train, Y_train = load_data(args.train)
    X_test, Y_test = load_data(args.test)
    classifier = MultiClass()
    classifier.fit(X_train, Y_train, X_test, Y_test, epochs=5)
    classifier.score(X_test, Y_test)
