import argparse
import tensorflow as tf
import numpy as np
from board import Board, GameStates

tf.logging.set_verbosity(tf.logging.INFO)

class GameModel():
    def __init__(self, board_size, learning_rate=0.1):
        self.board_size = board_size
        self.learning_rate = learning_rate

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(name='input', dtype='float32', input_shape=(self.board_size, self.board_size, 1,)))

        model.add(tf.layers.Conv2D(board_size**2, 3, padding='same', activation=tf.nn.relu, name='conv1'))
        model.add(tf.layers.MaxPooling2D(2, 2, padding='same', name='maxpool1'))

        model.add(tf.layers.Conv2D(model.get_layer(name='conv1').filters*2, 2, padding='same', activation=tf.nn.relu, name='conv2'))
        model.add(tf.layers.MaxPooling2D(2, 2, padding='same', name='maxpool2'))
        model.add(tf.layers.Dropout(0.5, name='dropout2'))
        model.add(tf.layers.BatchNormalization(name='batchnorm2'))

        model.add(tf.layers.Flatten())

        model.add(tf.layers.Dense(128, activation=tf.nn.tanh, name='fc3'))
        model.add(tf.layers.Dropout(0.5, name='dropout3'))
        model.add(tf.layers.BatchNormalization(name='batchnorm3'))

        model.add(tf.layers.Dense(4, name='output'))

        self.model = model

    def __call__(self, inputs):
        """
        Builds the model graph and returns it
        :param inputs: A batch of input game boards of size [m, board_size, board_size, 1]
        :return: A logits tensor of shape [m, 4] for <UP/DOWN/LEFT/RIGHT> action prediction
        """
        #X = tf.keras.Input(shape=(self.board_size, self.board_size, 1,), name='input', dtype='float32')
        #X = tf.keras.Input(name='input', dtype='float32', tensor=tf.cast(tf.reshape(inputs, self.input_shape), dtype='float32'))
        target_shape = tuple([-1] + list(self.model.layers[0].input_shape[1:]))
        #target_shape = self.model.layers[0].input_shape[1:]
        print("target_input_shape = ", target_shape)
        X = tf.cast(tf.reshape(inputs, target_shape), dtype='float32')
        return self.model(X)


def model_fn(features, labels, mode, params):
    model = GameModel(params['board_size'])
    boards = features

    labels = labels.reshape((-1, 4))

    print ("features = ", features.shape)
    print("labels = ", labels.shape)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate)

        # If we are running multi-GPU, we need to wrap the optimizer.
        if params.get('multi_gpu'):
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        logits = model(boards)
        print("logits.shape = ", logits.shape)
        print("labels.shape = ", labels.shape)
        loss = tf.losses.softmax_cross_entropy(labels, logits)
        accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(model.learning_rate, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    return None


def main(argv):
    flags = parser.parse_args()
    q_network = tf.estimator.Estimator(model_fn=model_fn,
                             model_dir=flags.model_dir,
                             params={'board_size': flags.size})

    gameboard = Board(flags.size)

    def train_input_fn():
        """Prepare data for training."""
        return (gameboard.board / gameboard.max_tile, np.array([1,0,0,0]).T)

    # Train and evaluate model.
    #for _ in range(flags.train_epochs // flags.epochs_between_evals):
        #mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)
    q_network.train(input_fn=train_input_fn)
        #eval_results = q_network.evaluate(input_fn=eval_input_fn)
        #print('\nEvaluation results:\n\t%s\n' % eval_results)

    # Export the model
    if flags.model_dir is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image,
        })
        #q_network.export_savedmodel(flags.model_dir, input_fn)
        q_network.export_savedmodel(flags.model_dir, input_fn)


parser = argparse.ArgumentParser(description='Create and train an AI to play the 2048 game using Reinforcement Learning with Tensorflow')
parser.add_argument('--size', metavar='<SIZE>', default=4, required=False, type=int, nargs=1, help='NxN board size for the 2048 board')
#parser.add_argument('--train_epochs', metavar='<NUM>', default=100, required=False, type=int, nargs=1, help='Number of epochs to train')
#parser.add_argument('--size', metavar='<SIZE>', default=4, required=False, type=int, nargs=1, help='NxN board size for the 2048 board')
parser.add_argument('--model_dir', metavar='<PATH>', default='./model', required=False, type=str, nargs=1, help='output directory for trained model')

if __name__ == "__main__":
    tf.app.run()