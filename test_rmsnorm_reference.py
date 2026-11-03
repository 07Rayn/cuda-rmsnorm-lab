import torch
from rmsnorm_reference import rmsnorm_reference
import torch.nn.functional as F
torch.manual_seed(42)
shapes = [
    (1,1),
    (1,7),
    (3,5),
    (8,128),
    (4,257),
]
for rows,cols in shapes:
    x_random = torch.randn(rows,cols,dtype=torch.float32)
    weight_random = torch.randn(cols,dtype=torch.float32)
    actual = rmsnorm_reference(x_random,weight_random)
    expected = F.rms_norm(x_random,normalized_shape=(cols,),weight=weight_random,eps=1e-6,)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    print(f"随机测试通过:shape=({rows},{cols})")