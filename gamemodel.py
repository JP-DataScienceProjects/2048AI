import numpy as np
import tensorflow as tf
import os

from gameboard import GameActions

class GameModel():
    def __init__(self, board_size, model_name, model_dir, learning_rate):
        self.board_size = board_size
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_file_path = self.model_dir + "q_network_model.json"
        self.weight_file_path = self.model_dir + "q_network_weights.h5py"
        self.learning_rate = learning_rate

        if os.path.exists(self.model_file_path):
            self.load_from_file()
        else:
            self.build_model()

    def build_model(self):

        with tf.variable_scope(self.model_name):
            #X = tf.layers.Input(name='X', dtype=tf.float32, shape=(self.board_size, self.board_size, 1,))
            X = tf.keras.layers.Input(name='X', dtype=tf.float32, shape=(self.board_size, self.board_size, 1,))

            #with tf.variable_scope("L1"):
            conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=tf.keras.activations.relu, name='conv1')(X)
            maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=1, padding='same', name='maxpool1')(conv1)

            conv2 = tf.keras.layers.Conv2D(filters=conv1.shape.dims[-1].value * 2, kernel_size=2, padding='same', activation=tf.keras.activations.relu, name='conv2')(maxpool1)
            maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=1, padding='valid', name='maxpool2')(conv2)

            #with tf.variable_scope("L2"):
            #flatten2 = tf.keras.layers.Flatten(name='flatten2')(maxpool2)
            #dropout2 = tf.keras.layers.Dropout(rate=0.4, name='dropout2')(flatten2)
            #batchnorm2 = tf.keras.layers.BatchNormalization(name='batchnorm2')(dropout2)

            #with tf.variable_scope("L3"):
            flatten2 = tf.keras.layers.Flatten(name='flatten2')(maxpool2)
            fc3 = tf.keras.layers.Dense(units=flatten2.shape.dims[-1].value // 4, activation=tf.keras.activations.relu, name='fc3')(flatten2)
            dropout3 = tf.keras.layers.Dropout(rate=0.5, name='dropout3')(fc3)
            batchnorm3 = tf.keras.layers.BatchNormalization(name='batchnorm3')(dropout3)

            fc4 = tf.keras.layers.Dense(units=max(batchnorm3.shape.dims[-1].value // 2, 32), activation=tf.keras.activations.relu, name='fc4')(batchnorm3)
            dropout4 = tf.keras.layers.Dropout(rate=0.5, name='dropout4')(fc4)
            batchnorm4 = tf.keras.layers.BatchNormalization(name='batchnorm4')(dropout4)

            #with tf.variable_scope("L4"):
            fc5 = tf.keras.layers.Dense(units=len(GameActions), name='fc5')(batchnorm4)
            X_action_mask = tf.keras.layers.Input(shape=(len(GameActions),), dtype=tf.float32, name='X_action_mask')
            output = tf.keras.layers.Multiply(name='output')([X_action_mask, fc5])

            self.model = tf.keras.Model(inputs=[X, X_action_mask], outputs=[output], name=self.model_name)
        self.compile()
        print("Built fresh model for {0}".format(self.model_name))

    def compile(self, learning_rate=None):
        lr = self.learning_rate if not learning_rate else learning_rate

        #self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=GameModel.clipped_loss)
        #self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.95), loss=GameModel.clipped_loss)
        #self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate), loss='mse')


        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=0.9, clipvalue=1.0), loss=GameModel.calc_loss)
        #self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate), loss=GameModel.clipped_loss)



        # model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=GameModel.clipped_loss)
        # model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.95), loss=tf.keras.losses.mean_squared_error)
        # model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.95), loss=GameModel.clipped_loss)

    def save_to_file(self):
        with open(self.model_file_path, 'w') as f:
            f.write(self.model.to_json())
        self.model.save_weights(self.weight_file_path, overwrite=True)
        print("Model/weights for {0} saved to {1} and {2}".format(self.model_name, self.model_file_path, self.weight_file_path))
        self.print_sample_weights()

    def load_from_file(self):
        with open(self.model_file_path) as f:
            self.model = tf.keras.models.model_from_json(f.read())
        self.model.load_weights(self.weight_file_path)
        self.compile()
        print("Model/weights for {0} loaded from {1} and {2}".format(self.model_name, self.model_file_path, self.weight_file_path))
        self.print_sample_weights()

    def print_sample_weights(self, samples=4):
        print("Sample weights for {0}".format(self.model_name))
        #for layer in [d for d in self.model.layers if isinstance(d, tf.keras.layers.Dense)]:
        for layer in [d for d in self.model.layers if isinstance(d, tf.keras.layers.Dense) or isinstance(d, tf.keras.layers.Conv2D)]:
            weights,biases = layer.get_weights()
            print("\t{0}: {1}, {2}".format(layer.name, np.ravel(weights)[:samples], biases[:samples]))

    def prepare_inputs(self, board_inputs, action_inputs=None):
        X = np.array(board_inputs).reshape((-1, self.board_size, self.board_size, 1))
        X_action_mask = action_inputs.reshape(-1, len(GameActions)) if isinstance(action_inputs, np.ndarray) else np.ones((X.shape[0], len(GameActions)))
        return (X, X_action_mask)

    def __call__(self, board_inputs, action_inputs=None):
        """
        Builds the model graph and returns it
        :param inputs: A batch of input game boards of size [m, board_size, board_size, 1]
        :return: A logits tensor of shape [m, 4] for <UP/DOWN/LEFT/RIGHT> action prediction
        """
        X, X_action_mask = self.prepare_inputs(board_inputs, action_inputs)
        return self.model.predict(x={'X': X, 'X_action_mask': X_action_mask})

    def copy_weights_to(self, newmodel):
        newmodel.model.set_weights(self.model.get_weights())
        newmodel.compile()
        return newmodel

    @staticmethod
    def calc_loss(y_true, y_pred):
        return tf.keras.backend.mean(tf.keras.backend.square(tf.keras.backend.sum(y_true - y_pred, axis=1)))
