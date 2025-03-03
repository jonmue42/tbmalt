import torch
from torch.autograd.functional import jacobian

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=15, sci_mode=False, linewidth=200, profile="full")

def fun(x):
    return x**2

x = torch.tensor([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0]], requires_grad=True)
#x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

print(jacobian(fun, x))
print(fun(x))
