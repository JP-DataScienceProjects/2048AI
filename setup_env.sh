#!/bin/sh
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-pip python3-tk screen
sudo apt install git-all
sudo curl -sSL https://get.docker.com/ | sh

sudo pip3 install --upgrade numpy h5py matplotlib tensorflow dill zope.event

git config --global user.email "john.pazzelli@gmail.com"
git config --global user.name "pazzelli"

