#!/bin/sh
#EXT_IP="$(curl ipinfo.io/ip)"
#echo $EXT_IP

rm model_checkpoints/events.out.tfevents.*
pkill tensorboard

sudo pip install tensorboard
tensorboard --host=0.0.0.0 --logdir=model_checkpoints --port 6006 &
#curl http://localhost:6006/

#start "" "http://localhost:5005/#distributions&run=."
