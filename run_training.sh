#!/bin/sh
./runTB.sh

lr=0.0007
max_tile=256

python3 tfmodel.py --train 10000 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr
python3 tfmodel.py --train 10000 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr
python3 tfmodel.py --train 10000 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr
python3 tfmodel.py --train 10000 --max_tile=$max_tile --debug --suppress_charts --learning_rate=$lr
python3 tfmodel.py --train 100000 --max_tile=$max_tile --debug --learning_rate=$lr
