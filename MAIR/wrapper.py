from seqeval.metrics import f1_score

from model import MultiClass_2H
from preprocessor import BertPreprocessor
from trainer import Trainer

import os


class MultiClass(object):
    def __init__(self, dropout_1=0.5, dropout_2=0.5, hidden_size_1=64, hidden_size_2=64, optimizer='adam'):
        self.model = None
        self.tagger = None

        self.p = BertPreprocessor()

        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.optimizer = optimizer

    def fit(self, x_train, y_train, x_valid=None, y_valid=None,
            epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        """Fit the model for a fixed number of epochs.
        Args:
            x_train: list of training model.
            y_train: list of training target (label) model.
            x_valid: list of validation model.
            y_valid: list of validation target (label) model.
            batch_size: Integer.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
            epochs: Integer. Number of epochs to train the model.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
            shuffle: Boolean (whether to shuffle the training model
                before each epoch). `shuffle` will default to True.
        """

        self.p.build(y_train)

        model = MultiClass_2H(sent_embedding_dim=768,
                          num_labels=self.p.label_size,
                          dropout_1=self.dropout_1,
                          dropout_2=self.dropout_2,
                          hidden_size_1=self.hidden_size_1,
                          hidden_size_2=self.hidden_size_2)

        model, loss = model.build()
        model.compile(loss=loss, optimizer=self.optimizer)

        trainer = Trainer(model, preprocessor=self.p)
        trainer.train(x_train, y_train, x_valid, y_valid,
                      epochs=epochs, batch_size=batch_size,
                      verbose=verbose, callbacks=callbacks,
                      shuffle=shuffle)

        if x_train and y_valid:
            self.model = trainer.best_model
            self.best_report = trainer.best_model_report
            print("Best model report: ")
            print(self.best_report)

    def score(self, x_test, y_test):
        """Returns the f1-micro score on the given test model and labels.
        Args:
            x_test : array-like, shape = (n_samples, sent_length)
            Test samples.
            y_test : array-like, shape = (n_samples, sent_length)
            True labels for x.
        Returns:
            score : float, f1-micro score.
        """
        if self.model:
            x_test = self.p.transform(x_test)
            lengths = map(len, y_test)
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred, lengths)
            score = f1_score(y_test, y_pred)
            return score
        else:
            raise OSError('Could not find a model. Call load(dir_path).')

    def predict(x_test):
        if self.model:
            x_test = self.p.transform(x_test)
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred, lengths)
            print(x_test)
            print(y_pred)


    def predict_to_iob(self, input_path, output_path):
        input_file = open(input_path, 'r')

        output_file = open(output_path, 'w')
        sentence, iob_lines, true_labels = [], [], []

        for input_line in input_file:
            if "DOCSTART CONFIG" in input_line or "DOCSTART FILE" in input_line:
                continue
            if input_line == '\n':
                x_test = self.p.transform([sentence])
                lengths = map(len, [true_labels])
                y_pred = self.model.predict(x_test)
                y_pred = self.p.inverse_transform(y_pred, lengths)
                for prediction, iob_line in zip(y_pred[0], iob_lines):
                    output_line = ''
                    for pos in iob_line:
                        output_line += pos + '\t'
                    output_line += prediction
                    output_file.write(output_line + '\n')
                output_file.write('\n')
                sentence = []
                iob_lines = []
            else:
                iob_line = input_line.split(sep='\t')[0:3]
                iob_lines.append(iob_line)
                sentence.append(input_line.split(sep='\t')[0])
                true_labels.append(input_line.split(sep='\t')[3])

    def predict_sentence(self, sentence):
        x_test = self.p.transform([sentence])
        lengths = [len(sentence)]
        y_pred = self.model.predict(x_test)
        y_pred = self.p.inverse_transform(y_pred, lengths)
        #print(y_pred)
        return y_pred[0]

    def save(self, model_path):
        weights_file = os.path.join(model_path, "weights.pkl")
        params_file = os.path.join(model_path, "params.pkl")
        preprocessor_file = os.path.join(model_path, "preprocessor.pkl")
        self.p.save(preprocessor_file)
        save_model(self.model, weights_file, params_file)

    @classmethod
    def load(cls, model_path, embedding_object):
        weights_file = os.path.join(model_path, "weights.pkl")
        params_file = os.path.join(model_path, "params.pkl")
        preprocessor_file = os.path.join(model_path, "preprocessor.pkl")
        self = cls(embedding_object)
        self.model = load_model(weights_file, params_file)
        self.p = VectorTransformer.load(preprocessor_file, embedding_object)
        return self

