SET lr=0.0007
SET max_tile=256

python tfmodel.py --train 10000 --max_tile=%max_tile% --debug --suppress_charts --learning_rate=%lr%
python tfmodel.py --train 10000 --max_tile=%max_tile% --debug --suppress_charts --learning_rate=%lr%
python tfmodel.py --train 10000 --max_tile=%max_tile% --debug --suppress_charts --learning_rate=%lr%
python tfmodel.py --train 10000 --max_tile=%max_tile% --debug --suppress_charts --learning_rate=%lr%
REM python tfmodel.py --train 4000 --max_tile=%max_tile% --debug --suppress_charts
REM python tfmodel.py --train 4000 --max_tile=%max_tile% --debug --suppress_charts
REM python tfmodel.py --train 4000 --max_tile=%max_tile% --debug --suppress_charts
python tfmodel.py --train 100000 --max_tile=%max_tile% --debug --learning_rate=%lr%