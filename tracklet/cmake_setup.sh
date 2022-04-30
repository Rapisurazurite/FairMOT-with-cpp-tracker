#!/bin/bash
#source /home/ubuntu/anaconda3/bin/activate Openpcdet
#export PATH=/usr/local/cuda-11.3/bin:$PATH
#echo $PATH
#
DIR=$1
EXTRA=${@:2}
set USE_NINJA=OFF
source /home/ubuntu/anaconda3/bin/activate zfk_MOT1.7
cd ${DIR} && python cmake_setup.py ${EXTRA}