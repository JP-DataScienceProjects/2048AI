import numpy as np
import tensorflow as tf

from gameboard import GameActions

class GameModel():
    def __init__(self, board_size, model_name, learning_rate):
        self.board_size = board_size
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):

        with tf.variable_scope(self.model_name):
            #X = tf.layers.Input(name='X', dtype=tf.float32, shape=(self.board_size, self.board_size, 1,))
            X = tf.keras.layers.Input(name='X', dtype=tf.float32, shape=(self.board_size, self.board_size, 1,))

            #with tf.variable_scope("L1"):
            #conv1 = tf.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.leaky_relu, name='conv1')(X)
            #maxpool1 = tf.layers.MaxPooling2D(pool_size=2, strides=1, padding='valid', name='maxpool1')(conv1)

            #conv2 = tf.layers.Conv2D(filters=conv1.shape.dims[-1].value * 2, kernel_size=2, padding='same', activation=tf.nn.leaky_relu, name='conv2')(maxpool1)
            #maxpool2 = tf.layers.MaxPooling2D(pool_size=2, strides=1, padding='valid', name='maxpool2')(conv2)

            #with tf.variable_scope("L2"):
            #flatten2 = tf.layers.Flatten(name='flatten2')(maxpool2)
            #dropout2 = tf.layers.Dropout(rate=0.4, name='dropout2')(flatten2)
            #batchnorm2 = tf.layers.BatchNormalization(name='batchnorm2')(dropout2)

            #with tf.variable_scope("L3"):
            flatten2 = tf.keras.layers.Flatten(name='flatten2')(X)
            fc3 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu, name='fc3')(flatten2)
            dropout3 = tf.keras.layers.Dropout(rate=0.5, name='dropout3')(fc3)
            batchnorm3 = tf.keras.layers.BatchNormalization(name='batchnorm3')(dropout3)

            fc4 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu, name='fc4')(batchnorm3)
            dropout4 = tf.keras.layers.Dropout(rate=0.5, name='dropout4')(fc4)
            batchnorm4 = tf.keras.layers.BatchNormalization(name='batchnorm4')(dropout4)

            #with tf.variable_scope("L4"):
            fc5 = tf.keras.layers.Dense(units=len(GameActions), name='fc5')(batchnorm4)
            X_action_mask = tf.keras.layers.Input(shape=(len(GameActions),), dtype=tf.float32, name='X_action_mask')
            output = tf.keras.layers.Multiply(name='output')([X_action_mask, fc5])

            model = tf.keras.Model(inputs=[X, X_action_mask], outputs=[output])
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=GameModel.clipped_loss)
            #model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.95), loss=tf.keras.losses.mean_squared_error)
            #model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.95), loss=GameModel.clipped_loss)
            #model.compile(optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate), loss=GameModel.clipped_loss)
            return model

    def prepare_inputs(self, board_inputs, action_inputs=None):
        X = board_inputs.reshape((-1, self.board_size, self.board_size, 1))
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
        return newmodel

    @staticmethod
    def clipped_loss(y_true, y_pred):
        # sq_err = tf.keras.backend.square(y_pred - y_true)
        # sq_err_clipped = tf.keras.backend.clip(sq_err, -1, 1)
        err_clipped = tf.keras.backend.clip(y_true - y_pred, -1, 1)
        sq_err_clipped = tf.keras.backend.square(err_clipped)
        total_err = tf.keras.backend.mean(tf.keras.backend.sum(sq_err_clipped, axis=-1))
        return total_err