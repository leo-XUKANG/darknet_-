# darknet_-
我只是想用ipad通过github看代码，没有任何侵犯原作者代码著作权的意思，我会在darknet代码上添加一下我自己的中文注释，看情况，时间有限，能写多少写多少

膜拜大神
下源码，请去官网，谢谢
![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).


1.depthwise convolution 实际上实现起来有两种方式,我曾想过优化卷积网络的运算方式,但是很多都失败了,效果不怎么好,
一些人,一般都是以空间换时间,比如少做一层的for循环,使用更大的矩阵
