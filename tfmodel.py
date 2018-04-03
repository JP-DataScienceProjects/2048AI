import os
import sys
from datetime import datetime
import itertools
import argparse
import random
import copy
from collections import deque
import pickle
import tensorflow as tf
from tensorflow import set_random_seed
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from gameboard import GameBoard, GameStates, GameActions

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


class GameTrainer():
    def __init__(self, board_size, save_model=True, model_dir=None, debug=True, learning_rate=0.001):
        self.save_model = save_model
        self.model_dir = model_dir
        self.weight_file_path = self.model_dir + "q_network_weights.h5py"
        self.experience_history_path = self.model_dir + "exp_history.p"
        self.debug = debug
        self.learning_rate = learning_rate

        self.q_network = GameModel(board_size, model_name='q_network', learning_rate=learning_rate)
        self.q_hat = self.q_network.copy_weights_to(GameModel(board_size, model_name='q_hat', learning_rate=learning_rate))
        print(self.q_network.model.summary())

    def calc_reward(self, oldboard, newboard):
        #reward = -newboard.max_tile if newboard.game_state == GameStates.LOSE else (newboard.max_tile if newboard.game_state == GameStates.WIN else newboard.score - oldboard.score)
        #nom_reward = (-newboard.max_tile if newboard.game_state == GameStates.LOSE else (newboard.max_tile if newboard.game_state == GameStates.WIN else newboard.score - oldboard.score))
        #if newboard.game_state == GameStates.LOSE: return -int(np.log2(newboard.max_tile))
        #if newboard.game_state == GameStates.WIN: return int(np.log2(newboard.max_tile))
        #return newboard.score - oldboard.score
        if newboard.game_state == GameStates.LOSE: return -1
        if newboard.game_state == GameStates.WIN: return 1
        return 0
        #if newboard.score == oldboard.score: return 0
        #return (int(np.log2(newboard.score - oldboard.score)) - 1) / (int(np.log2(newboard.max_tile)) - 1)


    def exec_action(self, gameboard, gameaction):
        oldboard = copy.deepcopy(gameboard)
        gameboard.make_move(gameaction)
        return (oldboard, self.calc_reward(oldboard, gameboard))

    def preprocess_state(self, gameboard):
        #return gameboard.board.astype(np.float32) / gameboard.max_tile
        return np.log2(np.clip(gameboard.board,1, gameboard.max_tile)) / np.log2(gameboard.max_tile)

    def select_action(self, gameboard, epsilon):
        if len(gameboard.action_set) <= 0: return None

        # Return a random action with probability epsilon, otherwise return the model's recommendation
        if np.random.rand() < epsilon: return random.sample(gameboard.action_set, 1)[0]

        # Return the action with highest probability that is actually possible on the board, as predicted by the Q network
        action_probs = self.q_network(self.preprocess_state(gameboard))
        best_actions = [GameActions(i) for i in np.argsort(np.ravel(action_probs))[::-1]]
        return next(a for a in best_actions if a in gameboard.action_set)

    def calculate_y_target(self, newboards, actions_oh, rewards, gamestates, gamma):
        # Calculate the target predictions based on the obtained rewards using the Q-hat network
        # The first two lines compute gamma times
        q_hat_output = self.q_hat(np.array(newboards))
        # print("Q-hat output values: ", q_hat_output)
        q_hat_pred = np.max(q_hat_output, axis=1)
        q_values = q_hat_pred * np.array([0 if s != GameStates.IN_PROGRESS.value else gamma for s in gamestates])
        total_reward = np.array(rewards + q_values)
        return actions_oh * total_reward.reshape((-1, 1))

    # def calculate_y_predicted(self, boards, t_actions):
    #     return self.q_network(np.array(boards), t_actions)

    def save_model_weights(self):
        # Export the weights from the Q-network when training completed
        if self.save_model and self.model_dir is not None:
            self.q_network.model.save_weights(self.weight_file_path)
            print("Q-network weights saved to " + self.weight_file_path)

    def restore_model_weights(self):
        # Attempt to restore Q-network weights from disk and copy to Q-hat, if present
        self.q_network.model.load_weights(self.weight_file_path, by_name=True)
        self.q_hat = self.q_network.copy_weights_to(self.q_hat)
        print("Loaded existing Q-network weights from " + self.weight_file_path)

    def save_experience_history(self, D):
        if os.path.exists(self.experience_history_path): os.remove(self.experience_history_path)
        saved = False
        while (not saved):
            try:
                pickle.dump(D, open(self.experience_history_path, "wb"))
                saved = True
                print("Saved gameplay experience to " + self.experience_history_path)
            except:
                print("WARNING: failed to save experience replay history.  Will try again...")

    def restore_experience_history(self):
        D = pickle.load(open(self.experience_history_path, "rb")) if os.path.exists(self.experience_history_path) and os.path.getsize(self.experience_history_path) > 0 else []
        if len(D) > 0: print("Restored gameplay experience from " + self.experience_history_path)
        return D

    def train_model(self, episodes=10, max_tile=2048, max_experience_history=700000, max_game_history=5000, max_epsilon=1.0, min_epsilon=0.1, mini_batch_size=32, gamma=0.99, update_qhat_weights_steps=10000):
        # Training variables
        D = self.restore_experience_history()  # experience replay queue
        gamehistory = deque(maxlen=max_game_history)    # history of completed games
        #max_epsilon = 1.0
        epsilon = max_epsilon       # probability of selecting a random action.  This is annealed from 1.0 to 0.1 over time
        update_frequency = 4        # Number of actions selected before the Q-network is updated again
        globalstep = 0

        approx_steps_per_episode = 200
        episodes_per_tb_output = 100
        steps_per_tb_output = approx_steps_per_episode * episodes_per_tb_output     # MUST BE A MULTIPLE OF update_frequency

        # Prepare a callback to write TensorBoard debugging output
        tbCallback = tf.keras.callbacks.TensorBoard(log_dir=self.model_dir, histogram_freq=1,
                                                    batch_size=mini_batch_size, write_graph=False,
                                                    write_images=False, write_grads=True)

        # Attempt to restore the model weights from the save file, if present
        try:
            self.restore_model_weights()
        except OSError:
            print("WARNING: no existing model weights available.  Will train a new model.\n")

        # Loop over requested number of games (episodes)
        loss = 0
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
                # Ensure history size is capped at max_history by randomly replacing an experience in the queue if necessary
                experience = (self.preprocess_state(oldboard), action.value, reward, self.preprocess_state(gameboard), gameboard.game_state.value)
                if len(D) >= max_experience_history: D[np.random.randint(0, len(D))] = experience
                else: D.append(experience)

                # Perform a gradient descent step on the Q-network when a game is finished or every so often
                if globalstep % update_frequency == 0 and len(D) >= mini_batch_size:
                    # Randomly sample from the experience history and unpack into separate arrays
                    batch = [D[i] for i in np.random.randint(0, len(D), mini_batch_size)]
                    oldboards, actions, rewards, newboards, gamestates = [list(k) for k in zip(*batch)]

                    # One-hot encode the actions for each of these boards as this will form the basis of the
                    # loss calculation
                    actions_one_hot = Utils.one_hot(actions, len(GameActions))

                    # Compute the target network output using the Q-hat network, actions and rewards for each
                    # sampled history item
                    y_target = self.calculate_y_target(newboards, actions_one_hot, rewards, gamestates, gamma)

                    # Perform a single gradient descent update step on the Q-network
                    callbacks = []
                    if self.debug and (globalstep % steps_per_tb_output == 0): callbacks.append(tbCallback)
                    X = [np.array(oldboards).reshape((-1, self.q_network.board_size, self.q_network.board_size, 1)), actions_one_hot]

                    if len(callbacks) > 0:
                        self.q_network.model.fit(x=X, y=y_target, validation_split=0.16, epochs=1, verbose=False, callbacks=callbacks)
                    else:
                        loss += self.q_network.model.train_on_batch(x=X, y=y_target)

                # Every so often, copy the network weights over from the Q-network to the Q-hat network
                # (this is required for network weight convergence)
                if globalstep % update_qhat_weights_steps == 0:
                    self.q_network.copy_weights_to(self.q_hat)
                    print("Weights copied to Q-hat network")
                    self.save_model_weights()

                # Perform annealing on epsilon
                epsilon = max(min_epsilon, min_epsilon + ((max_epsilon - min_epsilon) * (episodes - episode) / episodes))
                #epsilon = max_epsilon
                #if self.debug: print("epsilon: {:.4f}".format(epsilon))

            # Append metrics for each completed game to the game history list
            gameresult = (gameboard.game_state.value, gameboard.score, stepcount, gameboard.largest_tile_placed)
            if len(gamehistory) >= max_game_history: gamehistory.popleft()
            gamehistory.append(gameresult)
            print('result: {0}, score: {1}, steps: {2}, max tile: {3}'.format(GameStates(gameresult[0]).name, gameresult[1], gameresult[2], gameresult[3]))
            if (episode + 1) % 20 == 0:
                games_to_retrieve = min(len(gamehistory), 100)
                last_x_games = list(itertools.islice(gamehistory, len(gamehistory) - games_to_retrieve, None))
                last_x_results = list(zip(*last_x_games))[0]
                games_won = np.sum([1 if r == GameStates.WIN.value else 0 for r in last_x_results])
                print("\nEpisode {0}/{1}".format(episode + 1, episodes))
                print("Game win % (for last {:d} games): {:.1f}%".format(games_to_retrieve, 100. * (games_won / games_to_retrieve)))
                print("Epsilon = {:.3f}".format(epsilon))
                print("Training loss: {:.5f}\n".format(loss))
                loss = 0

            # Save experience history to disk periodically
            if self.save_model and (episode + 1) % 1000 == 0: self.save_experience_history(D)

        # Perform one final model weight save for next run
        if self.save_model:
            self.save_model_weights()
            self.save_experience_history(D)

        return gamehistory

    @staticmethod
    def display_training_history(gamehistory):
        results, scores, stepcounts, max_tiles = list(zip(*gamehistory))
        resultpercentages = np.cumsum([1 if r == GameStates.WIN.value else 0 for r in results]) / range(1, len(results) + 1)
        print("Final win %: {:2f}".format(resultpercentages[-1] * 100.))

        x = range(1, len(results) + 1)
        fig, ax_arr = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(5, 10))
        ax_arr[0].plot(x, resultpercentages)
        ax_arr[0].set_ylim(bottom=0)
        ax_arr[0].set_title('Win % (cumulative) = {:.2f}%'.format(resultpercentages[-1] * 100.))

        ax_arr[1].plot(x, scores)
        ax_arr[1].set_title('Score')

        ax_arr[2].plot(x, stepcounts)
        ax_arr[2].set_title('Actions per Game')

        ax_arr[3].plot(x, max_tiles)
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
    # Seed the RNGs
    seed = datetime.now().microsecond
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)

    flags = parser.parse_args()

    if flags.train:
        # Determine model checkpoint directory
        save_dir = os.getcwd() + os.sep + flags.model_dir + os.sep

        # Invoke the model trainer
        trainer = GameTrainer(board_size=flags.bsize, save_model=(not flags.no_save), model_dir=save_dir, debug=flags.debug, learning_rate=flags.learning_rate)
        game_history = trainer.train_model(episodes=flags.train, max_tile=flags.max_tile, min_epsilon=flags.min_epsilon, max_epsilon=flags.max_epsilon)
        if not flags.suppress_charts: GameTrainer.display_training_history(game_history)

    # Explicitly clear the keras session to avoid intermittent error message on termination
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create and train an AI to play the 2048 game using Reinforcement Learning with Tensorflow')
    parser.add_argument('--bsize', metavar='<SIZE>', default=4, required=False, type=int, help='NxN board size for the 2048 board')
    parser.add_argument('--train', metavar='<EPISODES>', default=10, required=False, type=int, help='# of training episodes (games) to play')
    parser.add_argument('--max_tile', metavar='<NUM>', default=2048, required=False, type=int, help='largest tile to use in the game')
    parser.add_argument('--suppress_charts', action='store_true', help='whether or not to suppress display of summary charts after training')
    parser.add_argument('--no_save', action='store_true', help='do not re-save the trained model')
    parser.add_argument('--debug', action='store_true', help='whether or not to write debug Tensorboard info')
    parser.add_argument('--model_dir', metavar='<PATH>', default='model_checkpoints', required=False, type=str, nargs=1,
                        help='output directory for trained model')
    parser.add_argument('--learning_rate', metavar='<NUM>', default=0.001, required=False, type=float, help='learning rate for optimizer')
    parser.add_argument('--min_epsilon', metavar='<NUM>', default=0.1, required=False, type=float,
                        help='minimum probability to select a random action')
    parser.add_argument('--max_epsilon', metavar='<NUM>', default=1.0, required=False, type=float,
                        help='maximum probability to select a random action')
    # parser.add_argument('--playonly', metavar='<SIZE>', default=False, required=False, type=bool, nargs=0,
    #                     help='observe gameplay using best model')

    #tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run()
    main(argv=sys.argv)
