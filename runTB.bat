pip install tensorboard
start "" "http://localhost:5005/#distributions&run=."
tensorboard --host=127.0.0.1 --port=5005 --logdir=model_checkpoints
