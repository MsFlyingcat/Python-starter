'''
交叉熵（Cross-Entropy）是一个衡量两个概率分布之间差异的指标，在机器学习中常常用来作为分类任务中模型输出的概率分布与真实标签之间的损失函数。
交叉熵损失函数有助于优化模型参数，使得模型的预测概率分布尽可能接近实际标签的分布。

假设在一个三分类问题中 (K = 3)，一个样本的真实标签为 [1, 0, 0]，表示该样本属于第一个类别。
模型对该样本的预测概率为[0.7, 0.2, 0.1]。那么，这个样本的交叉熵损失计算如下：
H(y,p)=−(1⋅log(0.7)+0⋅log(0.2)+0⋅log(0.1))
简化后得到：
H(y, p) = -\log(0.7)H(y,p)=−log(0.7)
'''

import numpy as np

def cross_entropy_loss(y_true, p_pred):
    # Ensure p_pred is a valid probability distribution
    assert np.isclose(np.sum(p_pred), 1), "Predicted probabilities must sum to 1."
    
    # Calculate the cross entropy loss
    loss = -np.sum(y_true * np.log(p_pred))
    return loss

# 示例数据
y_true = np.array([1, 0, 0])  # 真实标签
p_pred = np.array([0.7, 0.2, 0.1])  # 预测概率分布

# 计算交叉熵损失
loss = cross_entropy_loss(y_true, p_pred)
print(f"Cross-entropy loss: {loss}")

