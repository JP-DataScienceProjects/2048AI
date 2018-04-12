import os
import gc
import time
import itertools
import random
import copy
import shutil
import traceback
from collections import deque
import dill
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import Utils, RingBuf
from gameboard import GameBoard, GameStates, GameActions
from gamemodel import GameModel

class GameTrainer():
    def __init__(self, board_size, save_model=True, model_dir=None, debug=True, learning_rate=0.01, max_experience_history=600000):
        self.save_model = save_model
        self.model_dir = model_dir
        self.experience_history_path = self.model_dir + "exp_history.p"
        self.max_experience_history = max_experience_history
        self.debug = debug
        self.learning_rate = learning_rate

        self.q_network = GameModel(board_size, model_name='q_network', model_dir=self.model_dir, learning_rate=learning_rate)
        print(self.q_network.model.summary())
        self.q_hat = GameModel(board_size, model_name='q_hat', model_dir=self.model_dir, learning_rate=learning_rate)
        #self.q_hat = self.q_network.copy_weights_to(GameModel(board_size, model_name='q_hat', model_dir=self.model_dir, learning_rate=learning_rate))

    def calc_reward(self, oldboard, newboard):
        # The rewards should be normalized so that the max tile value can be changed without having to retrain
        if newboard.game_state == GameStates.LOSE: return -1
        nom_reward = np.clip((newboard.score - oldboard.score) // 2 , 1, None)
        reward = np.log2(nom_reward) / (np.log2(newboard.max_tile) - 1)
        return np.clip(reward, -1., 1.)
        # if newboard.game_state == GameStates.LOSE: return -1
        # if newboard.game_state == GameStates.WIN: return 1
        # return 0

    def exec_action(self, gameboard, gameaction):
        oldboard = copy.deepcopy(gameboard)
        gameboard.make_move(gameaction)
        return (oldboard, self.calc_reward(oldboard, gameboard))

    def preprocess_state(self, gameboard):
        #return gameboard.board.astype(np.float32) / gameboard.max_tile
        return (np.log2(np.clip(gameboard.board, 1, gameboard.max_tile))) / np.log2(gameboard.max_tile)

    def get_action_probabilities(self, gameboard):
        return np.ravel(self.q_network(self.preprocess_state(gameboard)))

    def select_action(self, gameboard, epsilon):
        if len(gameboard.action_set) <= 0: return None

        # Return a random action with probability epsilon, otherwise return the model's recommendation
        if np.random.rand() < epsilon: return random.sample(gameboard.action_set, 1)[0]

        # Return the action with highest probability that is actually possible on the board, as predicted by the Q network
        action_probs = self.get_action_probabilities(gameboard)
        best_actions = [GameActions(i) for i in np.argsort(action_probs)[::-1]]
        return next(a for a in best_actions if a in gameboard.action_set)

    def calculate_y_target(self, newboards, actions_oh, rewards, gamestates, gamma):
        # Determine best actions for the future board states from the current Q-network
        # (this is the DDQN approach for faster training)
        newboards_arr = np.array(newboards)
        best_actions = np.argmax(self.q_network(newboards_arr), axis=1)
        q_hat_output = self.q_hat(newboards_arr, Utils.one_hot(best_actions, len(GameActions)))
        #q_hat_output = self.q_hat(np.array(newboards))
        # print("Q-hat output values: ", q_hat_output)
        q_hat_pred = np.max(q_hat_output, axis=1)
        q_values = q_hat_pred * np.array([0 if s != GameStates.IN_PROGRESS.value else gamma for s in gamestates])
        total_reward = np.array(rewards) + q_values
        return actions_oh * total_reward.reshape((-1, 1))

    def save_experience_history(self, D):
        if os.path.exists(self.experience_history_path): os.remove(self.experience_history_path)
        saved = False
        f_hist = None
        while (not saved):
            try:
                f_hist = open(self.experience_history_path, "wb")
                dill.dump(D, f_hist)
                saved = True
                print("Saved gameplay experience to " + self.experience_history_path)
            except Exception as e:
                traceback.print_exc()
                print(e)
                print("WARNING: failed to save experience replay history.  Will try again in 5 seconds...")
                time.sleep(5)
            finally:
                if not f_hist is None and 'close' in dir(f_hist): f_hist.close()
        if os.path.getsize(self.experience_history_path) > 0: shutil.copy2(self.experience_history_path, self.experience_history_path.replace('.p', '_BACKUP.p'))

    def restore_experience_history(self):
        f_hist = open(self.experience_history_path, "rb") if os.path.exists(self.experience_history_path) and os.path.getsize(self.experience_history_path) > 0 else None
        if f_hist is None: return RingBuf(self.max_experience_history)
        D = dill.load(f_hist)
        f_hist.close()
        if isinstance(D, RingBuf) and len(D) > 0:
            print("Restored gameplay experience from " + self.experience_history_path)
            return D
        return RingBuf(self.max_experience_history)

    def train_model(self, episodes=10, max_tile=2048, max_game_history=500, max_epsilon=1.0, min_epsilon=0.1, mini_batch_size=32, gamma=0.99, update_qhat_weights_steps=10000):
        # Training variables
        D = self.restore_experience_history()  # experience replay queue
        gamehistory = deque(maxlen=max_game_history)    # history of completed games
        epsilon = max_epsilon       # probability of selecting a random action.  This is annealed from 1.0 to 0.1 over time
        update_frequency = 4        # Number of actions selected before the Q-network is updated again
        globalstep = 0

        # approx_steps_per_episode = 200
        # episodes_per_tb_output = 100
        # steps_per_tb_output = approx_steps_per_episode * episodes_per_tb_output     # MUST BE A MULTIPLE OF update_frequency

        # # Prepare a callback to write TensorBoard debugging output
        # tbCallback = tf.keras.callbacks.TensorBoard(log_dir=self.model_dir, histogram_freq=1,
        #                                             batch_size=mini_batch_size, write_graph=False,
        #                                             write_images=False, write_grads=True)

        q_hat_group = 1
        f_log = None
        if self.debug:
            f_log = open(self.model_dir + "training_log.csv", "w")
            f_log.write("Q_hat_group,Avg_Pred,Avg_Target,Avg_Diff,Won_Lost_Games,My_Loss,Actual_Loss\n")


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
                D.append(experience)

                # Perform a gradient descent step on the Q-network when a game is finished or every so often
                #if globalstep % update_frequency == 0 and len(D) >= mini_batch_size:
                if len(D) >= max(mini_batch_size, self.max_experience_history) and globalstep % update_frequency == 0:
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
                    # callbacks = []
                    # if self.debug and (globalstep % steps_per_tb_output == 0): callbacks.append(tbCallback)
                    X = [np.array(oldboards).reshape((-1, self.q_network.board_size, self.q_network.board_size, 1)), actions_one_hot]

                    # if len(callbacks) > 0:
                    #     self.q_network.model.fit(x=X, y=y_target, validation_split=0.16, epochs=1, verbose=False, callbacks=callbacks)
                    # else:
                    #loss += self.q_network.model.train_on_batch(x=X, y=y_target)


                    # print("Calling train_on_batch multiple times for batch of size: {0}".format(mini_batch_size))
                    # print("y_target = {0}\n".format(y_target))
                    # for i in range(10):
                    #     print("y_pred = {0}".format(np.vstack([np.ravel(self.q_network(oldboards[i], actions_one_hot[i])) for i in
                    #                                  range(mini_batch_size)])))
                    #     print("diff = {0}".format(np.vstack([np.ravel(self.q_network(oldboards[i], actions_one_hot[i])) for i in
                    #                                range(mini_batch_size)]) - y_target))
                    #     curloss = self.q_network.model.train_on_batch(x=X, y=y_target)
                    #     print("Current loss: {:.4f}\n".format(curloss))

                    write_log_entry = (globalstep % (update_frequency * 10) == 0) and f_log
                    if write_log_entry:
                        y_pred = self.q_network(oldboards, actions_one_hot)
                        avg_pred = np.mean(np.abs(np.sum(y_pred, axis=1)))
                        avg_target = np.mean(np.abs(np.sum(y_target, axis=1)))
                        diff = np.sum(y_target - y_pred, axis=1)
                        avg_diff = np.mean(np.abs(diff))
                        my_loss = np.mean(np.square(diff))
                        won_lost_games = np.sum(np.array(gamestates) < GameStates.IN_PROGRESS.value)

                    curloss = self.q_network.model.train_on_batch(x=X, y=y_target)
                    loss += curloss

                    if write_log_entry:
                        f_log.write("{:d},{:.7f},{:.7f},{:.7f},{:d},{:.7f},{:.7f}\n".format(q_hat_group, avg_pred, avg_target, avg_diff, won_lost_games, my_loss, curloss))


                # Every so often, copy the network weights over from the Q-network to the Q-hat network
                # (this is required for network weight convergence)
                if globalstep % update_qhat_weights_steps == 0:
                    # def gen_test_board():
                    #     testboard = GameBoard(self.q_network.board_size, max_tile)
                    #     testboard.board = np.random.randint(0, int(np.log2(max_tile)) + 1, self.q_network.board_size ** 2)
                    #     testboard.board = np.array([v if v <= 0 else np.power(2, v) for v in testboard.board])
                    #     return  self.preprocess_state(testboard)
                    def test_weight_update(testboard):
                        board_processed = self.preprocess_state(testboard)
                        q_res = np.ravel(self.q_network(board_processed))
                        q_hat_res = np.ravel(self.q_hat(board_processed))
                        print("\nQ-network result on testboard: {0}".format(q_res))
                        print("Q-hat result on testboard: {0}".format(q_hat_res))
                        print("Difference: {0}\n".format(q_res - q_hat_res))

                    # testboard = gen_test_board()
                    #testboard = oldboard
                    test_weight_update(oldboard)

                    self.q_network.copy_weights_to(self.q_hat)
                    print("Weights copied to Q-hat network")
                    self.q_network.save_to_file()

                    test_weight_update(oldboard)

                    lr_new = self.learning_rate / (1 + episode / (episodes/2))
                    #lr_new = self.learning_rate / np.sqrt(episode)
                    self.q_network.compile(learning_rate=lr_new)
                    print("Q-network learning rate updated to {:.6f}".format(lr_new))

                    q_hat_group += 1

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
                print("History queue {0}/{1}".format(len(D), self.max_experience_history))
                print("Game win % (for last {:d} games): {:.1f}%".format(games_to_retrieve, 100. * (games_won / games_to_retrieve)))
                print("Epsilon = {:.3f}".format(epsilon))
                print("Training loss: {:.5f}\n".format(loss))
                loss = 0

            # Save experience history to disk periodically
            if self.save_model and (episode + 1) % 2000 == 0:
                self.save_experience_history(D)
                # Output garbage
                print("GC.isenabled() = {0}".format(gc.isenabled()))
                print("Garbage:", gc.garbage)
                print("Counts:", gc.get_count())
                print("globals() = ", sorted(list(globals().keys())))

        # Perform one final model weight save for next run
        if self.save_model:
            self.q_network.save_to_file()
            self.save_experience_history(D)

        if f_log: f_log.close()

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
