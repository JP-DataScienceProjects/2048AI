#!/bin/sh

python3 tfmodel.py --train 10 --max_tile 256 --debug --learning_rate=0.0007
#python3 tfmodel.py --train 100 --max_tile 256 --debug --no_save --learning_rate=0.0007 --max_epsilon=0.0 --min_epsilon=0.0
#python3 -m cProfile  tfmodel.py --train 10 --max_tile 256 --debug --suppress_charts --learning_rate=0.0007 > result.txt