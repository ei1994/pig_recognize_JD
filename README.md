## pig_recognize_JD
[京东算法组猪脸识别官网](http://jddjr.jd.com/item/4 "京东算法")    

## 1、比赛总结：
**时间：** 2017.10 — 2017.12
* （1）**训练集的制作：** 对每头猪的视频，每隔10帧取一帧图像保存。观察分析测试数据集，考虑训练数据和测试数据的图像分布要一致的原则，所以将获取的图像帧中的猪位置截取出来。采用方法：先人工标注获取的图像帧数据集中的一小部分图像，标注出猪的位置信息，然后利用其训练YOLOv2用于识别猪的位置（只需要识别出图像中猪的位置，不需要确定猪的类别，由于图像拍摄环境不是很复杂，识别目标位置对YOLOv2来说很容易）。利用训练好的YOLOv2网络预测识别剩余图像帧数据集中猪的位置并截取出猪目标图像，然后筛选去除一些切割错误的图像，将最后制作成tfrecord格式的数据，这样训练数据集就完成了。后期为提升识别准确率，将猪脸也加入到训练集中训练。 
* （2）**数据预处理：** 数据扩充等相关处理。
* （3）**网络训练：** 基础网络选择Inception-ResNet-v2，softmax交叉熵损失函数（要区别30头猪），batchsize是28，优化器选择adam，学习率0.01，训练一段时间后，降为0.0001。在训练这一阶段，主要是调节网络优化器及参数，观察是否过拟合等，选择训练最好的模型。
* （4）**预测及后处理：** 网络预测一张图片，得到的是30个类别对于的概率值。根据得分的计算公式分析可以看出，一张图像对正确类别的预测概率越低，计算的logloss增长越大，所以相对于让30个类别中，正确类别的预测概率更高，如果让错误类别的预测概率不那么低更重要，因为总有图像预测错误，将对应正确的类别概率预测的很低（比如一张正确类别为第一头猪的图像，标签为1，预测标签为1得到的概率为0.9，即使网络再进一步以0.95的概率肯定是标签为1，则计算图像logloss减少0.0540（log0.95-log0.9）；但如果网络预测不准确，预测该图像标签为1得到的是0.2，如果进一步将概率增大到0.25，计算图像logloss减少0.22314（log0.25-log0.2））。基于此，我们尽量让预测的类别标签概率值不出现极端情况，也就是让网络不那么肯定是哪头猪（如：某头猪预测概率为0.99就不可取），从而换取整体logloss的降低。实验证明该方案在一定程度上缺失提升了排名。
     
## 2、比赛排名：
![](/京东.PNG "京东猪脸识别")



