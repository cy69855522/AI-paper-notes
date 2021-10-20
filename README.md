# AI-paper-notes
S: sequence, C: channel, FC: fully connected layer, Q: query, K: key, V: value

## Attention Is All You Need
<details>
  <summary>Self-Attention 时间复杂度S^2 · C</summary>
  -	给出输入X [S, C]
  -	通过3个FC将X线性变换为Q [S, C], K [S, C], V [S, C]
  -	对于S中的每个元素s，用对应的Ks与所有序列元素的V求点积，然后通过SoftMax得到s对S中每个元素的注意力。
  -	注意求完点积后要除以根号C进行缩放再过SoftMax，原因如下：
  	假设Q K中向量的元素都是相互独立的均值为 0，方差为 1 的随机变量，点积的均值为0，方差为C。若某个点积过大会导致其余点积在SoftMax处的梯度很小，不利于网络收敛。
  -	利用注意力加权V求和得到s的响应，最后用FC对响应做线性变换得到输出
  ```python
  # Given X [S, C]
  Q, K, V = fc1(X), fc2(X), fc3(X)  # [S, C]
  A = Q @ K.T  # [S, S]
  A = (A / √C).softmax(dim=-1)  # [S, S]
  R = A @ V  # [S, C]
  O = fc4(R)  # [S, C]
  ```

</details>

## An Attention Free Transformer
AFT-simple 时间复杂度 S · C
-	给出输入X [S, C]
-	通过3个FC将X线性变换为Q [S, C], K [S, C], V [S, C]
-	直接对K做SoftMax得到全局注意力图，并加权求和V得到全局响应。对于每个s，利用对应的Qs对全局响应做通道维度的缩放得到独特的注意力图。
```python
# Given X [S, C]
Q, K, V = fc1(X), fc2(X), fc3(X)  # [S, C]
A_global = K.softmax(dim=0)  # [S, C]
R_global = (A_global * V).sum(dim=0)  # [C]
R = Q.sigmoid() * R_global  # [S, C]
O = fc4(R)  # [S, C]
```

AFT-full 时间复杂度 S^2 · C
-	增加了可学习的参数W [S, S]作为位置偏差
-	对K做SoftMax前先加了W，其余部分同AFT-simple
  - W使用了重参技巧：`W = u[S, 128] @ v[128, S]`
```python
# Given X [S, C], W [S, S]
Q, K, V = fc1(X), fc2(X), fc3(X)  # [S, C]
R = W.exp() @ (K.exp() * V) / W.exp() @ K.exp()  # [S, C]
R = Q.sigmoid() * R  # [S, C]
O = fc4(R)  # [S, C]
```

AFT-local 时间复杂度 S^2 · C
- 只训练W中的一部分，其余权重固定为0
```python
# Given X [S, C], W [S, S]
for i in range(S):
  for j in range(S):
    if abs(i-j) >= K
Q, K, V = fc1(X), fc2(X), fc3(X)  # [S, C]
R = W.exp() @ (K.exp() * V) / W.exp() @ K.exp()  # [S, C]
R = Q.sigmoid() * R  # [S, C]
O = fc4(R)  # [S, C]
```

AFT-conv 时间复杂度 S · k · C
- 略
