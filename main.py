import os
import sys
from datetime import datetime
import argparse
import random
import zope.event
import tensorflow as tf
from tensorflow import set_random_seed
import numpy as np

from utils import Utils
from gameboard import GameBoard, GameStates, GameActions, OnBoardChanged
from gamegrid import GameGrid
from gametrainer import GameTrainer



def run_interactive(board_size, model_dir):
    trainer = GameTrainer(board_size=board_size, model_dir=model_dir)
    trainer.restore_model_weights()

    def on_board_updated(event):
        if not isinstance(event, OnBoardChanged): return
        probs = trainer.get_action_probabilities(event.board)
        msg = " ".join(["{:s}: {:.1f}%".format(GameActions(i).name, p) for i,p in enumerate(Utils.softmax(probs) * 100.)])
        #msg = " ".join(["{:s}: {:.1f}%".format(GameActions(i).name, p) for i, p in enumerate(probs)])
        print(msg)

    zope.event.subscribers.append(on_board_updated)
    gamegrid = GameGrid()



def main(argv):
    # Seed the RNGs
    seed = datetime.now().microsecond
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)

    flags = parser.parse_args()
    # Determine model checkpoint directory
    save_dir = os.getcwd() + os.sep + flags.model_dir + os.sep

    if flags.train:
        # Invoke the model trainer
        trainer = GameTrainer(board_size=flags.bsize, save_model=(not flags.no_save), model_dir=save_dir, debug=flags.debug, learning_rate=flags.learning_rate)
        game_history = trainer.train_model(episodes=flags.train, max_tile=flags.max_tile, min_epsilon=flags.min_epsilon, max_epsilon=flags.max_epsilon)
        if not flags.suppress_charts: GameTrainer.display_training_history(game_history)

    else:
        run_interactive(board_size=flags.bsize, model_dir=save_dir)


    # Explicitly clear the keras session to avoid intermittent error message on termination
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create and train an AI to play the 2048 game using Reinforcement Learning with Tensorflow')
    parser.add_argument('--bsize', metavar='<SIZE>', default=4, type=int, help='NxN board size for the 2048 board')
    parser.add_argument('--train', metavar='<EPISODES>', type=int, help='# of training episodes (games) to play')
    parser.add_argument('--max_tile', metavar='<NUM>', default=2048, type=int, help='largest tile to use in the game')
    parser.add_argument('--suppress_charts', action='store_true', help='whether or not to suppress display of summary charts after training')
    parser.add_argument('--no_save', action='store_true', help='do not re-save the trained model')
    parser.add_argument('--debug', action='store_true', help='whether or not to write debug Tensorboard info')
    parser.add_argument('--model_dir', metavar='<PATH>', default='model_checkpoints', type=str, nargs=1,
                        help='directory to read/write the trained model')
    parser.add_argument('--learning_rate', metavar='<NUM>', default=0.001, type=float, help='learning rate for optimizer')
    parser.add_argument('--min_epsilon', metavar='<NUM>', default=0.1, type=float,
                        help='minimum probability to select a random action')
    parser.add_argument('--max_epsilon', metavar='<NUM>', default=1.0, type=float,
                        help='maximum probability to select a random action')
    # parser.add_argument('--playonly', metavar='<SIZE>', default=False, required=False, type=bool, nargs=0,
    #                     help='observe gameplay using best model')

    #tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run()
    main(argv=sys.argv)
