#!/bin/sh
#./runTB.sh

lr=0.001
max_tile=128
min_epsilon=0.1
max_epsilon=1.0

python3 main.py --train 10001 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon --max_epsilon=$max_epsilon
python3 main.py --train 10002 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon --max_epsilon=$max_epsilon
python3 main.py --train 10003 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon --max_epsilon=$max_epsilon
python3 main.py --train 10004 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon --max_epsilon=$max_epsilon
python3 main.py --train 10005 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon --max_epsilon=$max_epsilon
#python3 main.py --train 200005 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon --max_epsilon=$max_epsilon
#python3 main.py --train 200006 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon --max_epsilon=$max_epsilon
#python3 main.py --train 200007 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon --max_epsilon=$max_epsilon
#python3 main.py --train 200008 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon --max_epsilon=$max_epsilon
#python3 main.py --train 200009 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon --max_epsilon=$max_epsilon
#python3 main.py --train 200010 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr --min_epsilon=$min_epsilon --max_epsilon=$max_epsilon
