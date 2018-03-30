#!/bin/sh
./runTB.sh

lr=0.0007
max_tile=256
min_epsilon=0.5

python3 tfmodel.py --train 200000 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon
python3 tfmodel.py --train 200000 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon
python3 tfmodel.py --train 200000 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon
python3 tfmodel.py --train 200000 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon
python3 tfmodel.py --train 200000 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon
python3 tfmodel.py --train 200000 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon
python3 tfmodel.py --train 200000 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon
#python3 tfmodel.py --train 200000 --max_tile=$max_tile --debug --learning_rate=$lr
