"""
Model definition.
"""
import json

from keras.layers import Dense, Input, Dropout
from keras.models import Model, model_from_json


def save_model(model, weights_file, params_file):
    with open(params_file, 'w') as f:
        params = model.to_json()
        json.dump(json.loads(params), f, sort_keys=True, indent=4)
        model.save_weights(weights_file)


def load_model(weights_file, params_file):
    with open(params_file) as f:
        model = model_from_json(f.read(), custom_objects={'CRF': CRF})
        model.load_weights(weights_file)

    return model


class MultiClass_2H(object):
    """
    Multiclass sentence classificator with 2 hidden layers 

    """

    def __init__(self,
                 num_labels,
                 sent_embedding_dim=100,
                 hidden_size_1=64,
                 hidden_size_2=64,
                 dropout_1=0.5,
                 dropout_2=0.5):
        """Build a NN with two hidden layers
        Args:
            num_labels (int): number of labels.
            sent_embedding_dim (int): sent2vec embedding length.
            hidden_size_1 (int): first hidden fully-connected layer size.
            hidden_size_2 (int): second hidden fully-connected layer size.
            dropout_1 (float): dropout rate after first hidden layer.
            dropout_2 (float): dropout rate after second hidden layer.
        """

        super(MultiClass_2H).__init__()
        self._num_labels = num_labels
        self._sent_embedding_dim = sent_embedding_dim
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2


    def build(self):
        input_layer = Input(batch_shape=(None, self._sent_embedding_dim), dtype='float32', name='sentence_input')
        inputs = [input_layer]
        
        hidden_layer_1 = Dense(self.hidden_size_1, activation='tanh')(input_layer)
        dropout_1 = Dropout(self.dropout_1)(hidden_layer_1)
        
        hidden_layer_2 = Dense(self.hidden_size_2, activation='tanh')(dropout_1)
        dropout_2 = Dropout(self.dropout_2)(hidden_layer_2)

        output_layer = Dense(self._num_labels, activation='softmax')(hidden_layer_2)

        loss = 'categorical_crossentropy'
        model = Model(inputs=inputs, outputs=output_layer)

        print(model.summary())

        return model, loss

