# FairMOT with cpp tracker

## 主要修改
将FairMOT的多目标跟踪使用的卡尔曼滤波器改为cpp实现，
输入输出效果与原来的一致，提升了追踪算法运行的速度。

## 使用

项目使用pytorch extension实现，不过运算没有用到torch矩阵。
使用的是Eigen库，因此需要安装pytorch以及Eigen。
Eigen安装：`sudo apt install libeigen3-dev`
```shell
cd tracklet
conda activate ${your_env}
python setup.py install
```
将会安装一个名为tracklet的包。调用的话将原有的调用kalman_filter.py替换成如下即可。
```python
# from tracking_utils.kalman_filter import KalmanFilter
from tracklet.kalman_filter import KalmanFilter
```

## 效果

原使用demo.py对示例视频追踪，使用yolo检测器，推理20ms，追踪算法运行10ms。
修改后，追踪结果不变， 追踪算法运行5ms。

## 下一步

目前只修改卡尔曼滤波器部分，下一步进行更上层的修改。

## 原项目
https://github.com/ifzhang/FairMOT