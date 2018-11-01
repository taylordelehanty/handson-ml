import tensorflow as tf
import numpy as np

def fetch_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

class ArtificialNeuralNet():
    def __init__(self, X_, y_):
        tf.reset_default_graph()
        self.X = X_
        self.y = y_ if y_.shape[1] else y_.reshape(-1, 1)
        self.inputs = tf.placeholder(tf.float32, (None, self.X.shape[1]), name='X')
        self.labels = tf.placeholder(tf.float32, (None, self.y.shape[1]), name='y')
        self.layers = [self.inputs]
    
    def add_dense_layer(self, n_inputs, n_neurons, activation, name):
        layer = tf.layers.dense(self.layers[-1], n_neurons, activation=activation, name=name)
        self.layers.append(layer)
    
    def calculate_loss(self):
        pass
    
    def train_network(self, epochs, batch_size, learning_rate):
        for layer_metadata in self.network_metadata():
            new_layer = tf.layers.dense(**layer_metadata)
            self.layers.append(layer)