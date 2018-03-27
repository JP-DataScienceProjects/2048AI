import os
import sys
import argparse
import random
import copy
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from gameboard import GameBoard, GameStates, GameActions

class GameModel():
    def __init__(self, board_size, model_name):
        self.board_size = board_size
        self.model_name = model_name
        self.model = self.build_model()

    def build_model(self):

        with tf.variable_scope(self.model_name):
            X = tf.layers.Input(name='X', dtype=tf.float32, shape=(self.board_size, self.board_size, 1,))

            with tf.variable_scope("L1"):
                conv1 = tf.layers.Conv2D(filters=self.board_size ** 2, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')(X)
                maxpool1 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpool1')(conv1)

                #conv2 = tf.layers.Conv2D(filters=conv1.shape.dims[-1].value * 2, kernel_size=2, padding='same', activation=tf.nn.relu, name='conv2')(maxpool1)
                #maxpool2 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpool2')(conv2)

            with tf.variable_scope("L2"):
                flatten2 = tf.layers.Flatten()(maxpool1)
                dropout2 = tf.layers.Dropout(rate=0.5, name='dropout2')(flatten2)
                batchnorm2 = tf.layers.BatchNormalization(name='batchnorm2')(dropout2)

            with tf.variable_scope("L3"):
                fc3 = tf.layers.Dense(units=32, activation=tf.nn.tanh, name='fc3')(batchnorm2)
                dropout3 = tf.layers.Dropout(rate=0.5, name='dropout3')(fc3)
                batchnorm3 = tf.layers.BatchNormalization(name='batchnorm3')(dropout3)

            with tf.variable_scope("L4"):
                fc4 = tf.layers.Dense(units=4, name='fc4')(batchnorm3)
                X_action_mask = tf.keras.Input(shape=(len(GameActions),), dtype=tf.float32, name='X_action_mask')
                output = tf.keras.layers.Multiply()([X_action_mask, fc4])

            model = tf.keras.Model(inputs=[X, X_action_mask], outputs=[output])
            model.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_error)

            #print(model.summary())
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

    def copy_weights(self, newmodel):
        newmodel.model.set_weights(self.model.get_weights())
        return newmodel


class GameTrainer():
    def __init__(self, board_size, save_model=True, model_dir=None, debug=True, mini_batch_size=32, gamma=0.99, reset_qhat_batches=3):
        self.q_network = GameModel(board_size, model_name='q_network')
        self.q_hat = self.q_network.copy_weights(GameModel(board_size, model_name='q_hat'))
        self.save_model = save_model
        self.model_dir = model_dir
        self.debug = debug
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.reset_qhat_batches = reset_qhat_batches

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

        # Return the action with highest probability that is actually possible on the board, as predicted by the Q network
        action_probs = self.q_network(self.preprocess_state(gameboard))
        best_actions = [GameActions(i) for i in np.argsort(np.ravel(action_probs))[::-1]]
        return next(a for a in best_actions if a in gameboard.action_set)

    def calculate_y_target(self, newboards, actions_oh, rewards, gamestates):
        # Calculate the target predictions based on the obtained rewards using the Q-hat network
        # The first two lines compute gamma times
        q_hat_pred = np.max(self.q_hat(np.array(newboards)), axis=1)
        q_values = q_hat_pred * np.array([0 if s != GameStates.IN_PROGRESS.value else self.gamma for s in gamestates])
        total_reward = np.array(rewards) + q_values
        return actions_oh * total_reward.reshape((-1, 1))

    # def calculate_y_predicted(self, boards, t_actions):
    #     return self.q_network(np.array(boards), t_actions)


    def train_model(self, episodes=10, max_tile=2048, max_history=50000, min_epsilon=0.1):
        # Attempt to restore Q-network weights from disk and copy to Q-hat, if present
        if self.save_model:
            weight_file = self.model_dir + "q_network_weights.h5py"
            tbCallback = tf.keras.callbacks.TensorBoard(log_dir=self.model_dir, histogram_freq=0, batch_size=self.mini_batch_size, write_graph=True, write_images=True)

            try:
                self.q_network.model.load_weights(weight_file)
                self.q_network.copy_weights(self.q_hat)
                print("Loaded existing Q-network weights from " + weight_file)
            except OSError:
                print("WARNING: no existing model weights available.  Will train a new model.\n")


        # Training variables
        D = []              # experience replay queue
        gamehistory = []    # history of completed games
        epsilon = 1.0       # probability of selecting a random action.  This is annealed from 1.0 to 0.1 over time
        globalstep = 0

        for episode in range(episodes):
            # New game
            gameboard = GameBoard(self.q_network.board_size, max_tile=max_tile)

            # Play the game
            stepcount = 0
            while gameboard.game_state == GameStates.IN_PROGRESS:
                stepcount += 1
                globalstep += 1

                # Select an action to perform.  It will be a random action with probability epsilon, otherwise
                # the action with highest probability from the Q-network will be chosen
                action = self.select_action(gameboard, epsilon)
                oldboard, reward = self.exec_action(gameboard, action)

                # Append the (preprocessed) original board, selected action, reward and new board to the history
                # This is to implement experience replay for reinforcement learning
                D.append((self.preprocess_state(oldboard), action, reward, self.preprocess_state(gameboard), gameboard.game_state.value))

                # Ensure history size is capped at max_history.  Remove excess history at random so a mix of
                # old and new experience is retained
                if len(D) > max_history:
                    for i in np.random.choice(len(D), len(D) - max_history, replace=False): D.pop(i)

                # Perform a gradient descent step on the Q-network when a game is finished or every so often
                if stepcount % self.mini_batch_size == 0 or gameboard.game_state != GameStates.IN_PROGRESS:
                    # Randomly sample from the experience history and unpack into separate arrays
                    batch = [D[i] for i in np.random.choice(len(D), self.mini_batch_size, replace=False)]
                    oldboards, actions, rewards, newboards, gamestates = [list(k) for k in zip(*batch)]

                    # t_actions = tf.keras.backend.one_hot([a.value for a in actions], len(GameActions))
                    #t_rewards = tf.reshape(tf.constant(rewards, dtype=tf.float32), shape=[len(rewards), 1])
                    #t_rewards = tf.keras.backend.reshape(rewards, shape=(len(rewards), 1))

                    # One-hot encode the actions for each of these boards as this will form the basis of the
                    # loss calculation
                    actions_one_hot = Utils.one_hot([a.value for a in actions], len(GameActions))

                    # Compute the target network output using the Q-hat network, actions and rewards for each
                    # sampled history item
                    y_target = self.calculate_y_target(newboards, actions_one_hot, rewards, gamestates)
                    #y_pred = self.calculate_y_predicted(oldboards, t_actions)

                    #loss = tf.losses.mean_squared_error(labels=y_target, predictions=y_pred)
                    #loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y_pred)
                    #loss = tf.squared_difference(y_pred, y_target)
                    #loss = tf.losses.mean_squared_error(labels=y_target, predictions=y_pred)

                    # if not graph_initialized:
                    #     #self.session.run(tf.global_variables_initializer())
                    #     tf.keras.backend.get_session().run(tf.global_variables_initializer())
                    #     graph_initialized = True

                    #self.q_network.model.train_on_batch(x=[np.array(oldboards), t_actions], y=y_target)

                    # actions_one_hot = np.zeros((len(oldboards), len(GameActions)))
                    # actions_one_hot[np.arange(len(oldboards)), np.array([a.value for a in actions])] = 1

                    # Perform a single gradient descent update step on the Q-network
                    callbacks = []
                    if self.debug and (globalstep % (20 * self.mini_batch_size) == 0): callbacks.append(tbCallback)
                    self.q_network.model.fit(x={'X': np.array(oldboards).reshape((-1, self.q_network.board_size, self.q_network.board_size, 1)), 'X_action_mask': actions_one_hot}, y=y_target, epochs=1, verbose=1, callbacks=callbacks)


                    #_, loss_value = self.session.run((optimizer.minimize(loss), loss))
                    #_, loss_value = self.session.run((optimizer.minimize(loss), loss))

                # Every so often, copy the network weights over from the Q-network to the Q-hat network
                # (this is required for network weight convergence)
                if globalstep % (self.reset_qhat_batches * self.mini_batch_size) == 0:
                    self.q_hat.model.set_weights(self.q_network.model.get_weights())

                # Perform annealing on epsilon
                epsilon = max(min_epsilon, min_epsilon + ((1.0 - min_epsilon) * (max_history - len(D)) / max_history))

            # Append metrics for each completed game to the game history list
            gamehistory.append({'result': gameboard.game_state.name, 'score': gameboard.score, 'steps': stepcount, 'max tile placed': gameboard.largest_tile_placed})

        # Export the weights from the Q-network when training completed
        if self.save_model and self.model_dir is not None:
            self.q_network.model.save_weights(weight_file)
            print ("Q-network weights saved to " + weight_file)

        return gamehistory

    @staticmethod
    def display_training_history(gamehistory):
        results, scores, stepcounts, max_tiles = list(zip(*[g.values() for g in gamehistory]))
        resultpercentages = np.cumsum([1 if r == GameStates.WIN.name else 0 for r in results]) / len(results)

        x = range(1, len(results) + 1)
        fig, ax_arr = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(5, 10))
        ax_arr[0].plot(x, resultpercentages)
        ax_arr[0].set_ylim([0,1])
        ax_arr[0].set_title('Win % (cumulative)')

        ax_arr[1].plot(x, scores)
        #ax_arr[1].set_ylim([0, 1])
        ax_arr[1].set_title('Score')

        ax_arr[2].plot(x, stepcounts)
        # ax_arr[1].set_ylim([0, 1])
        ax_arr[2].set_title('Actions per Game')

        ax_arr[3].plot(x, max_tiles)
        # ax_arr[1].set_ylim([0, 1])
        ax_arr[3].set_title('Max Tile Placed')

        ax_arr[3].set_xlabel("Game #")
        ax_arr[3].xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.canvas.set_window_title("2048 Game Training Results")
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()

        return gamehistory

class Utils():
    @staticmethod
    def one_hot(data, classes):
        encoded = np.zeros((len(data), classes))
        encoded[np.arange(encoded.shape[0]), np.array(data)] = 1
        return encoded


def main(argv):
    flags = parser.parse_args()

    if flags.train:
        # Determine model checkpoint directory
        save_dir = os.getcwd() + os.sep + flags.model_dir + os.sep

        # Invoke the model trainer
        trainer = GameTrainer(flags.bsize, flags.save, save_dir, flags.debug)
        game_history = trainer.train_model(episodes=flags.train)
        GameTrainer.display_training_history(game_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create and train an AI to play the 2048 game using Reinforcement Learning with Tensorflow')
    parser.add_argument('--bsize', metavar='<SIZE>', default=4, required=False, type=int, nargs=1, help='NxN board size for the 2048 board')
    parser.add_argument('--train', metavar='<EPISODES>', default=10, required=False, type=int, nargs=1, help='# of training episodes (games) to play')
    parser.add_argument('--save', action='store_true', default=True, help='whether or not to save the trained model')
    parser.add_argument('--debug', action='store_true', default=True, help='whether or not to write debug Tensorboard info')
    parser.add_argument('--model_dir', metavar='<PATH>', default='model_checkpoints', required=False, type=str, nargs=1,
                        help='output directory for trained model')
    # parser.add_argument('--train_epochs', metavar='<NUM>', default=100, required=False, type=int, nargs=1, help='Number of epochs to train')
    # parser.add_argument('--playonly', metavar='<SIZE>', default=False, required=False, type=bool, nargs=0,
    #                     help='observe gameplay using best model')

    #tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run()
    main(argv=sys.argv)
