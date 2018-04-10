#!/bin/sh
#./runTB.sh

lr=0.007
max_tile=256

python3 main.py --train 100 --max_tile=$max_tile --debug --learning_rate=$lr --suppress_charts
python3 main.py --train 100 --max_tile=$max_tile --debug --learning_rate=$lr --suppress_charts
#python3 main.py --train 100 --max_tile 256 --debug --no_save --learning_rate=0.0007 --max_epsilon=0.0 --min_epsilon=0.0
#python3 -m cProfile  main.py --train 10 --max_tile 256 --debug --suppress_charts --learning_rate=0.0007 > result.txt
