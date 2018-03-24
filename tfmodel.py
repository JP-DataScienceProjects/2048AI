import sys
import argparse
import random
import copy
import tensorflow as tf
import numpy as np
from board import Board, GameStates, GameActions

sess = tf.Session()

class GameModel():
    def __init__(self, board_size):
        self.board_size = board_size
        self.model = self.build_model(board_size)

    def build_model(self, board_size):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(name='input', dtype='float32',
                                             input_shape=(self.board_size, self.board_size, 1,)))

        model.add(tf.layers.Conv2D(board_size ** 2, 3, padding='same', activation=tf.nn.relu, name='conv1'))
        model.add(tf.layers.MaxPooling2D(2, 2, padding='same', name='maxpool1'))

        model.add(tf.layers.Conv2D(model.get_layer(name='conv1').filters * 2, 2, padding='same', activation=tf.nn.relu,
                                   name='conv2'))
        model.add(tf.layers.MaxPooling2D(2, 2, padding='same', name='maxpool2'))
        model.add(tf.layers.Dropout(0.5, name='dropout2'))
        model.add(tf.layers.BatchNormalization(name='batchnorm2'))

        model.add(tf.layers.Flatten())

        model.add(tf.layers.Dense(64, activation=tf.nn.tanh, name='fc3'))
        model.add(tf.layers.Dropout(0.5, name='dropout3'))
        model.add(tf.layers.BatchNormalization(name='batchnorm3'))

        model.add(tf.layers.Dense(4, name='output'))
        return model

    def __call__(self, inputs):
        """
        Builds the model graph and returns it
        :param inputs: A batch of input game boards of size [m, board_size, board_size, 1]
        :return: A logits tensor of shape [m, 4] for <UP/DOWN/LEFT/RIGHT> action prediction
        """
        target_shape = tuple([-1] + list(self.model.layers[0].input_shape[1:]))
        X = tf.cast(tf.reshape(inputs, target_shape), dtype='float32')
        return self.model(X)

    def __copy__(self):
        newmodel = GameModel(self.board_size)
        newmodel.model.set_weights(self.model.get_weights())
        return newmodel


def calc_reward(oldboard, newboard):
    return newboard.score - oldboard.score

def exec_action(gameboard, gameaction):
    oldboard = copy.deepcopy(gameboard)
    return (oldboard, calc_reward(oldboard, gameboard.make_move(gameaction)))

def preprocess_state(gameboard):
    return gameboard.board / gameboard.max_tile

def select_action(gameboard, model, epsilon):
    if len(gameboard.action_set) <= 0: return None

    # Return a random action with probability epsilon, otherwise return the model's recommendation
    if np.random.rand() < epsilon: return random.sample(gameboard.action_set, 1)
    return forward_propagate(model, gameboard)

def forward_propagate(model, gameboard):
    # Forward propagate the board through the policy network and return
    # the first best action that is actually possible on the current board
    #action_probs = tf.nn.softmax(model.predict(preprocess_state(gameboard)), axis=1)
    action_probs = sess.run(tf.nn.softmax(model(preprocess_state(gameboard)), axis=1))
    best_actions = [GameActions(i) for i in (np.argsort(action_probs)[::-1])]
    return next(a for a in best_actions if a in gameboard.action_set)

def encode_action(action):
    res = np.zeros(len(GameActions))
    res[action.value] = 1
    return res

def train_model(model_dir, q_network, episodes=1, mini_batch_size=32, learning_rate=0.01, epsilon=0.05, gamma=0.99, reset_qhat_steps=20):
    D = []
    history = []

    #q_hat = copy.deepcopy(q_network)
    q_hat = q_network.__copy__()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    for episode in range(episodes):
        gameboard = Board(q_network.board_size, max_tile=2048)

        # Play the game
        stepcount = 0
        while gameboard.game_state == GameStates.IN_PROGRESS:
            stepcount += 1
            action = select_action(gameboard, q_network, epsilon)
            oldboard, reward = exec_action(gameboard, action)
            D.append((preprocess_state(oldboard), action, reward, preprocess_state(gameboard)))

            if gameboard.game_state != GameStates.IN_PROGRESS or len(D) >= mini_batch_size:
                indices = np.random.randint(0, len(D), min(mini_batch_size, len(D)))
                oldboards, actions, rewards, newboards = [list(k) for k in zip(*D[indices])]
                y = [r + (0 if newboards[i].game_state != GameStates.IN_PROGRESS else gamma * encode_action(forward_propagate(q_hat, newboards[i]))) for i,r in enumerate(rewards)]

                #q_network.train(input_fn=lambda: (oldboards, y), steps=1)
                loss = tf.losses.mean_squared_error(labels=y, predictions=q_network(oldboards))
                _, loss_value = sess.run((optimizer.minimize(loss), loss))

            if stepcount % reset_qhat_steps == 0: q_hat = q_network.__copy__()

        history.append({'steps': stepcount, 'score': gameboard.score, 'result': gameboard.game_state.name})

    # # Export the model
    # if model_dir is not None:
    #     board_placeholder = tf.placeholder(tf.float32, [None, board_size, board_size, 1])
    #     input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
    #         'board_size': board_placeholder,
    #     })
    #     # q_network.export_savedmodel(flags.model_dir, input_fn)
    #     q_network.export_savedmodel(model_dir, input_fn)


    # def prepare_training_inputs():
    #     """Prepare data for training."""
    #     return (preprocess_state(gameboard), np.array([1, 0, 0, 0]).T)

    # Train and evaluate model.
    #for _ in range(flags.train_epochs // flags.epochs_between_evals):
        #mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)
    # q_network.train(input_fn=prepare_training_inputs)
        #eval_results = q_network.evaluate(input_fn=eval_input_fn)
        #print('\nEvaluation results:\n\t%s\n' % eval_results)

    return history


def model_fn(features, labels, mode, params):
    model = GameModel(params['board_size'])
    boards = features

    labels = labels.reshape((-1, len(GameActions)))

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(boards)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions
            # export_outputs={
            #     'classify': tf.estimator.export.PredictOutput(predictions)
            # }
            )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate)

        # # If we are running multi-GPU, we need to wrap the optimizer.
        # if params.get('multi_gpu'):
        #     optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        logits = model(boards)
        loss = tf.losses.softmax_cross_entropy(labels, logits)
        #accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(model.learning_rate, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        #tf.identity(accuracy[1], name='train_accuracy')

        ## Save accuracy scalar to Tensorboard output.
        #tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    return None


def main(argv):
    flags = parser.parse_args()
    # q_network = tf.estimator.Estimator(model_fn=model_fn,
    #                          model_dir=flags.model_dir,
    #                          params={'board_size': flags.bsize})

    #gameboard = Board(flags.bsize)

    sess.run(tf.global_variables_initializer())
    q_network = GameModel(flags.bsize)
    with sess:
        game_history = train_model(flags.model_dir, q_network, flags.bsize)

    print(game_history)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create and train an AI to play the 2048 game using Reinforcement Learning with Tensorflow')
    parser.add_argument('--bsize', metavar='<SIZE>', default=4, required=False, type=int, nargs=1,
                        help='NxN board size for the 2048 board')
    parser.add_argument('--learning_rate', metavar='<VAL>', default=0.1, required=False, type=int, nargs=1,
                        help='learning rate to use when training')
    # parser.add_argument('--train_epochs', metavar='<NUM>', default=100, required=False, type=int, nargs=1, help='Number of epochs to train')
    # parser.add_argument('--playonly', metavar='<SIZE>', default=False, required=False, type=bool, nargs=0,
    #                     help='observe gameplay using best model')
    parser.add_argument('--model_dir', metavar='<PATH>', default='./model', required=False, type=str, nargs=1,
                        help='output directory for trained model')

    tf.logging.set_verbosity(tf.logging.INFO)
    #tf.app.run()
    main(argv=sys.argv)
