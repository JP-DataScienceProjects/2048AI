SET lr=0.0007
SET max_tile=256
python tfmodel.py --train 10 --max_tile=%max_tile% --debug --learning_rate=%lr%
REM python tfmodel.py --train 100 --max_tile=%max_tile% --debug --no_save --learning_rate=%lr% --max_epsilon=0.0 --min_epsilon=0.0
REM python -m cProfile  tfmodel.py --train 10 --max_tile 256 --debug --suppress_charts --learning_rate=0.07 > result.txt