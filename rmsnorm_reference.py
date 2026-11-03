import torch
def rmsnorm_reference(x,weight,eps=1e-6):
    normalized =  x * torch.rsqrt((x*x).mean(dim=-1,keepdim=True) + eps)
    return normalized * weight
x = torch.tensor([
    [1.0,2.0,3.0,4.0],
    [2.0,0.0,0.0,0.0],
],dtype=torch.float32)
weight = torch.tensor([1.0,2.0,1.0,0.5])
y = rmsnorm_reference(x,weight)
expected = torch.tensor(
    [
        [0.365148, 1.460593, 1.095445, 0.730297],
        [1.999999, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)

torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)
print("测试一通过：与手算结果一致。")
x_zeros = torch.zeros((2,4),dtype=torch.float32)
y_zero = rmsnorm_reference(x_zeros,weight)
torch.testing.assert_close(y_zero, x_zeros, rtol=0, atol=0)
print("测试二通过：全零输入得到全零输出。")
