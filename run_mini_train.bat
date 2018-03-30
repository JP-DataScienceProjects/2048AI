python tfmodel.py --train 10 --max_tile 256 --debug --learning_rate=0.07
REM python tfmodel.py --train 100 --max_tile 256 --debug --no_save --learning_rate=0.07 --max_epsilon=0.0 --min_epsilon=0.0
REM python -m cProfile  tfmodel.py --train 10 --max_tile 256 --debug --suppress_charts --learning_rate=0.07 > result.txt