# 一、引出Attention

 - 1. 从人类视觉到计算机理解图像
   2. `卷积+池化`——计算机理解
   3. `注意力机制`——更有效的理解方式

# 二、介绍Attention

- 1. 介绍`Attention`的计算组成: `Query、Key和Value`，交代Attention的输入为`序列`
  2. 理解三者的关系，阐述VIT-Attention方式的原理就是`Self Attention`
  3. 简单交代`Local Attention`与`Global Attention`的区别
  4. 简单阐明`VIT-Attention`中为什么更多地选择`Local Attention`
  5. 利用前面2理解的Attention计算关系，到具体Vision中`如何构造Attention需要的序列数据`
  6. 交代本次教程主要针对的VIT模型复现安排:`VIT`、`Swin Transformer`、`Focal Transformer`.