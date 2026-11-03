import torch

# 两行输入，每行四个元素。先在 CPU 上观察计算过程。
x = torch.tensor(
    [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)

# 每一列有一个权重，两行共用这组权重。
weight = torch.tensor([1.0, 2.0, 1.0, 0.5])
eps = 1e-6

squared = x * x
mean_square = squared.mean(dim=-1, keepdim=True)
rstd = torch.rsqrt(mean_square + eps)
normalized = x * rstd
y = normalized * weight

print("输入 x:\n", x)
print("平方:\n", squared)
print("每行平方的均值:\n", mean_square)
print("每行的缩放系数 rstd:\n", rstd)
print("归一化结果:\n", normalized)
print("乘权重后的输出 y:\n", y)
# 根据这个例子的手算结果，检查输出。
expected = torch.tensor(
    [
        [0.365148, 1.460593, 1.095445, 0.730297],
        [1.999999, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)

torch.testing.assert_close(
    y,
    expected,
    rtol=1e-5,
    atol=1e-6,
)

print("检查通过：输出与手算结果一致。")