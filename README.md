# <center>TensorFlow 2 实现手写数字识别和猫狗二分类识别</center>

# 1 手写数字识别
（1）训练模型  
数据集：采用TensorFlow2.3的TensorFlow-datasets内置的mnist数据集 ，必须是2.1版本。  
> pip install Tensorflow_datasets==2.1 
神经网络：全连接  
损失函数：分类交叉熵  
优化器：Adam  
评价准则：准确率  
Epochs：100  
（2）测试模型  
测试集：采用TensorFlow2.3的TensorFlow-datasets内置的mnist数据集，形状改为28*28  
（3）环境  
> Python 3.8  
TensorFlow 2,3  
Tensorflow_datasets 2.1  
scipy  
numpy  
PIL  
matplotlib  
scipy  

# 2 猫狗分类
（1）训练模型  
数据集：采用kaggle公开的猫狗数据集，文件名称是cat.**.jpg或dog.**.jpg  
神经网络：卷积神经网络  
损失函数：分类交叉熵  
优化器：Adam  
评价准则：准确率  
Epochs：100  
（2）测试模型  
测试集：采用kaggle公开的猫狗测试集，没有标签  
（3）环境  
> python 3.8  
Tensorflow 2.3  
numpy   
pandas  
matplotlib  
cv2  

<img src="https://cdn.nlark.com/yuque/0/2021/png/1780216/1625292906904-d43bb33e-2ebc-46cf-a54b-1a992c21ac69.png?x-oss-process=image%2Fresize%2Cw_399" width="50%">
