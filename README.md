# FDU cv 第一次作业
本次代码使用方式：
将3Layer_NN.py、weight.txt放于python环境中，自行将data文件夹放于同一目录下，运行文件即可启动 

打开的文件的语句为
with open(f"data/cifar-10-batches-py/data_batch_{i}", 'rb') as f:  # 使用python的标准输入库读取文件
请保证data文件夹与源代码在同一父目录下
## 实现了自动参数查找与手动输入参数训练模型的功能  
在运行文件后    
输入：1 即可进入自动参数查找模式，程序将从      
    hidden_sizes = [128]   
    learning_rates = [0.01, 0.05]   
    batch_sizes = [64, 128]   
    reg_lambdas = [0.01, 0.1]  
    的参数空间中进行20次迭代的训练与验证，最后得到其中最优的参数，并绘制不同超参数下的迭代折线对比图   
    同时，会将搜索到的最佳权重储存到result.txt文件中
输入：2 可进入手动输入参数训练，根据提示输入参数后，将对给定超参数进行训练与验证，并返回训练结果 

输入：3 可进入自设置权重模式，该模式下将直接读入同目录下result.txt文件中的权重并设置模型，返回该模型下的准确率与loss
