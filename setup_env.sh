#!/bin/sh
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-pip
sudo apt-get install python3-tk
sudo apt install git-all

sudo pip3 install --upgrade numpy h5py matplotlib tensorflow

git config --global user.email "john.pazzelli@gmail.com"
git config --global user.name "pazzelli"

