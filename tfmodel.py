import sys
import argparse
import random
import copy
import tensorflow as tf
import numpy as np
from board import Board, GameStates, GameActions

class GameModel():
    def __init__(self, board_size, model_name):
        self.board_size = board_size
        self.model_name = model_name
        self.model = self.build_model()

    def build_model(self):

        with tf.variable_scope(self.model_name):
            X = tf.layers.Input(name='X', dtype=tf.float32, shape=(self.board_size, self.board_size, 1,))
            #X = tf.placeholder(name='X', dtype=tf.float32, shape=(None, self.board_size, self.board_size, 1))

            #model.add(tf.keras.layers.InputLayer(name='input', dtype='float32', input_shape=(board_size, board_size, 1,)))

            conv1 = tf.layers.Conv2D(filters=self.board_size ** 2, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')(X)
            maxpool1 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpool1')(conv1)

            #conv2 = tf.layers.Conv2D(filters=conv1.shape.dims[-1].value * 2, kernel_size=2, padding='same', activation=tf.nn.relu, name='conv2')(maxpool1)
            #maxpool2 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpool2')(conv2)

            flatten2 = tf.layers.Flatten()(maxpool1)
            dropout2 = tf.layers.Dropout(rate=0.5, name='dropout2')(flatten2)
            batchnorm2 = tf.layers.BatchNormalization(name='batchnorm2')(dropout2)

            fc3 = tf.layers.Dense(units=32, activation=tf.nn.tanh, name='fc3')(batchnorm2)
            dropout3 = tf.layers.Dropout(rate=0.5, name='dropout3')(fc3)
            batchnorm3 = tf.layers.BatchNormalization(name='batchnorm3')(dropout3)

            fc4 = tf.layers.Dense(units=4, name='fc4')(batchnorm3)
            #t_output = tf.keras.backend.placeholder(dtype=tf.float32, shape=(-1, len(GameActions)), name='output')

            X_action_mask = tf.keras.Input(shape=(len(GameActions),), dtype=tf.float32, name='X_action_mask')

            output = tf.keras.layers.Multiply()([X_action_mask, fc4])

            model = tf.keras.Model(inputs=[X, X_action_mask], outputs=[output])
            model.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_error)

            print(model.summary())
            return model

    def __call__(self, board_inputs, action_inputs=None):
        """
        Builds the model graph and returns it
        :param inputs: A batch of input game boards of size [m, board_size, board_size, 1]
        :return: A logits tensor of shape [m, 4] for <UP/DOWN/LEFT/RIGHT> action prediction
        """
        #target_shape = tuple([-1] + list(self.model.layers[0].input_shape[1:]))
        #X = tf.cast(tf.reshape(inputs, target_shape), dtype='float32')

        #return self.model(tf.cast(tf.reshape(inputs, (-1, self.board_size, self.board_size, 1)), dtype=tf.float32))

        #return self.model.predict(tf.reshape(inputs, target_shape))
        #reshaped = tf.keras.layers.Reshape(target_shape)(inputs)
        #return self.model.predict([board_inputs.reshape((-1, self.board_size, self.board_size, 1)), action_inputs])

        #X = tf.Constant(value=board_inputs.reshape((-1, self.board_size, self.board_size, 1)), dtype=tf.float32)
        X = board_inputs.reshape((-1, self.board_size, self.board_size, 1))
        #X_action_mask = tf.Constant(value=action_inputs.reshape(-1, len(GameActions)) if isinstance(action_inputs, np.ndarray) else np.ones((X.shape[0], len(GameActions)), dtype=np.float32), dtype=tf.float32)
        X_action_mask = action_inputs.reshape(-1, len(GameActions)) if isinstance(action_inputs, np.ndarray) else np.ones((X.shape[0], len(GameActions)))
        return self.model.predict(x={'X': X, 'X_action_mask': X_action_mask}, verbose=True)

    # def __copy__(self):
    #     newmodel = GameModel(self.board_size)
    #     newmodel.model.set_weights(self.model.get_weights())
    #     return newmodel
    def copy_weights(self, newmodel):
        newmodel.model.set_weights(self.model.get_weights())
        return newmodel


class GameTrainer():
    def __init__(self, board_size, model_dir=None, mini_batch_size=32, learning_rate=0.01, gamma=0.99, reset_qhat_steps=20):
        self.q_network = GameModel(board_size, model_name='q_network')
        self.q_hat = self.q_network.copy_weights(GameModel(board_size, model_name='q_hat'))
        self.session = tf.Session()
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qhat_steps = reset_qhat_steps

    def calc_reward(self, oldboard, newboard):
        return newboard.score - oldboard.score

    def exec_action(self, gameboard, gameaction):
        oldboard = copy.deepcopy(gameboard)
        gameboard.make_move(gameaction)
        return (oldboard, self.calc_reward(oldboard, gameboard))

    def preprocess_state(self, gameboard):
        return gameboard.board.astype(np.float32) / gameboard.max_tile

    def select_action(self, gameboard, epsilon):
        if len(gameboard.action_set) <= 0: return None

        # Return a random action with probability epsilon, otherwise return the model's recommendation
        if np.random.rand() < epsilon: return random.sample(gameboard.action_set, 1)[0]

        #action_probs = self.forward_propagate(self.q_network, self.preprocess_state(gameboard))
        action_probs = self.q_network(self.preprocess_state(gameboard))
        #action_probs = self.session.run(self.q_network.model, feed_dict={'X': self.preprocess_state(gameboard)})
        best_actions = [GameActions(i) for i in np.argsort(np.ravel(action_probs))[::-1]]
        return next(a for a in best_actions if a in gameboard.action_set)

    def calculate_y_target(self, boards, actions, rewards, gamestates):
        q_hat_pred = np.max(self.q_hat(np.array(boards)), axis=1, keepdims=True)
        q_values = q_hat_pred * np.array([0 if s != GameStates.IN_PROGRESS else 1 for s in gamestates]).reshape((len(boards),1)) * self.gamma
        total_reward = np.array(rewards).reshape((len(boards), 1)) + q_values

        y_target = np.zeros((len(boards), len(GameActions)))
        y_target[np.arange(len(boards)), np.array([a.value for a in actions])] = 1
        return y_target * total_reward

        #q_values = tf.keras.backend.max(x=q_hat_pred, axis=1, keepdims=True, name='q_hat_values') * self.gamma
        #total_reward = tf.add(t_rewards, tf.where(tf.keras.backend.equal([g.value for g in gamestates], GameStates.IN_PROGRESS.value), q_values, tf.zeros(q_values.shape)), name='total_reward')
        #return tf.multiply(t_actions, total_reward, name='y_target')
        #return tf.keras.layers.multiply(inputs=[t_actions, total_reward])

    def calculate_y_predicted(self, boards, t_actions):
        #y_pred = np.zeros((len(oldboards), len(GameActions)))
        #for i in range(len(y_pred)): y_pred[i, actions[i].value] = 1
        # y_pred *= forward_propagate(q_network, np.array(oldboards))

        #return tf.multiply(t_actions, self.q_network.model(np.array(boards)), name='y_predicted')
        return self.q_network(np.array(boards), t_actions)

    # def encode_action(action):
    #     res = np.zeros(len(GameActions))
    #     res[action.value] = 1
    #     return res


    def train_model(self, episodes=10):
        D = []
        gamehistory = []
        epsilon = 0.

        #q_hat = q_network.__copy__()
        tf.keras.backend.get_session().run(tf.global_variables_initializer())

        #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        graph_initialized = False
        for episode in range(episodes):
            gameboard = Board(self.q_network.board_size, max_tile=2048)

            # Play the game
            stepcount = 0
            while gameboard.game_state == GameStates.IN_PROGRESS:
                stepcount += 1
                action = self.select_action(gameboard, epsilon)
                oldboard, reward = self.exec_action(gameboard, action)
                D.append((self.preprocess_state(oldboard), action, reward, self.preprocess_state(gameboard), gameboard.game_state))

                if gameboard.game_state != GameStates.IN_PROGRESS or len(D) >= self.mini_batch_size:
                    batch = [D[i] for i in np.random.choice(len(D), self.mini_batch_size, replace=False)]
                    oldboards, actions, rewards, newboards, gamestates = [list(k) for k in zip(*batch)]

                    t_actions = tf.keras.backend.one_hot([a.value for a in actions], len(GameActions))
                    #t_rewards = tf.reshape(tf.constant(rewards, dtype=tf.float32), shape=[len(rewards), 1])
                    #t_rewards = tf.keras.backend.reshape(rewards, shape=(len(rewards), 1))

                    y_target = self.calculate_y_target(newboards, actions, rewards, gamestates)
                    #y_pred = self.calculate_y_predicted(oldboards, t_actions)

                    #loss = tf.losses.mean_squared_error(labels=y_target, predictions=y_pred)
                    #loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y_pred)
                    #loss = tf.squared_difference(y_pred, y_target)
                    #loss = tf.losses.mean_squared_error(labels=y_target, predictions=y_pred)

                    if not graph_initialized:
                        #self.session.run(tf.global_variables_initializer())
                        tf.keras.backend.get_session().run(tf.global_variables_initializer())
                        graph_initialized = True

                    #self.q_network.model.train_on_batch(x=[np.array(oldboards), t_actions], y=y_target)

                    actions_one_hot = np.zeros((len(oldboards), len(GameActions)))
                    actions_one_hot[np.arange(len(oldboards)), np.array([a.value for a in actions])] = 1

                    self.q_network.model.fit(x={'X': np.array(oldboards).reshape((-1, 4, 4, 1)), 'X_action_mask': actions_one_hot}, y=y_target, epochs=1, verbose=1)


                    #_, loss_value = self.session.run((optimizer.minimize(loss), loss))
                    #_, loss_value = self.session.run((optimizer.minimize(loss), loss))

                if stepcount % self.reset_qhat_steps == 0: self.q_hat.model.set_weights(self.q_network.model.get_weights())

            gamehistory.append({'steps': stepcount, 'score': gameboard.score, 'result': gameboard.game_state.name})
            epsilon = max(epsilon - 0.01, 0.1)

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
        # for _ in range(flags.train_epochs // flags.epochs_between_evals):
        # mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)
        # q_network.train(input_fn=prepare_training_inputs)
        # eval_results = q_network.evaluate(input_fn=eval_input_fn)
        # print('\nEvaluation results:\n\t%s\n' % eval_results)

        return gamehistory


    # def model_fn(features, labels, mode, params):
    #     model = GameModel(params['board_size'])
    #     boards = features
    #
    #     labels = labels.reshape((-1, len(GameActions)))
    #
    #     if mode == tf.estimator.ModeKeys.PREDICT:
    #         logits = model(boards)
    #         predictions = {
    #             'classes': tf.argmax(logits, axis=1),
    #             'probabilities': tf.nn.softmax(logits),
    #         }
    #         return tf.estimator.EstimatorSpec(
    #             mode=tf.estimator.ModeKeys.PREDICT,
    #             predictions=predictions
    #             # export_outputs={
    #             #     'classify': tf.estimator.export.PredictOutput(predictions)
    #             # }
    #         )
    #
    #     if mode == tf.estimator.ModeKeys.TRAIN:
    #         optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate)
    #
    #         # # If we are running multi-GPU, we need to wrap the optimizer.
    #         # if params.get('multi_gpu'):
    #         #     optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    #
    #         logits = model(boards)
    #         loss = tf.losses.softmax_cross_entropy(labels, logits)
    #         # accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))
    #
    #         # Name tensors to be logged with LoggingTensorHook.
    #         tf.identity(model.learning_rate, 'learning_rate')
    #         tf.identity(loss, 'cross_entropy')
    #         # tf.identity(accuracy[1], name='train_accuracy')
    #
    #         ## Save accuracy scalar to Tensorboard output.
    #         # tf.summary.scalar('train_accuracy', accuracy[1])
    #
    #         return tf.estimator.EstimatorSpec(
    #             mode=tf.estimator.ModeKeys.TRAIN,
    #             loss=loss,
    #             train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
    #
    #     return None


def main(argv):
    flags = parser.parse_args()
    # q_network = tf.estimator.Estimator(model_fn=model_fn,
    #                          model_dir=flags.model_dir,
    #                          params={'board_size': flags.bsize})

    # gameboard = Board(flags.bsize)

    #sess.run(tf.global_variables_initializer())
    #q_network = GameModel(flags.bsize)
    trainer = GameTrainer(flags.bsize, flags.model_dir)
    game_history = trainer.train_model()

    #with sess:
        #game_history = train_model(flags.model_dir, q_network, flags.bsize)

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
    # tf.app.run()
    main(argv=sys.argv)
