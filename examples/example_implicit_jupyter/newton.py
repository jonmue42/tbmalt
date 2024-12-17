import torch
import torch.nn as nn

class PointLayer_grad(nn.Module):
    def __init__(self, out_features, tol = 1e-4, max_iter = 50):
        super().__init__()
        #self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        #print('Beginning forward')
        #print(x)
        z = torch.zeros_like(x)
        #print(z)
        self.iterations = 0

        while self.iterations < self.max_iter:
            z_next = torch.sqrt(z + x)
            g = z - z_next
            self.err = torch.norm(g)
            if self.err < self.tol:
                break

            #newton step
            J = 1 - 1/(2 * z_next)
            z = z_next
            self.iterations += 1
        
        z = torch.sqrt(z + x)
        #z.register_hook(lambda grad : 1)
        return z


class PointLayer_nograd(nn.Module):
    def __init__(self, out_features, tol = 1e-4, max_iter = 50):
        super().__init__()
        #self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        #print('Beginning forward')
        #print(x)
        z = torch.zeros_like(x)
        #print(z)
        self.iterations = 0

        with torch.no_grad():
            while self.iterations < self.max_iter:
                z_next = torch.sqrt(z + x)
                g = z - z_next
                self.err = torch.norm(g)
                if self.err < self.tol:
                    break
                z = z_next
                self.iterations += 1

        z = torch.sqrt(z + x)
        #z.register_hook(lambda grad : grad / (1/(0.5 * z)) )
        return z


layer_nograd = PointLayer_nograd(1)
layer_grad = PointLayer_grad(1)
X = torch.tensor([[1.0]], requires_grad=True)
print(f"Input X: {X}")
Z_nograd = layer_nograd(X)
Z_grad = layer_grad(X)

print(f"Output Z_grad: {Z_grad}")
print(f"Terminated after {layer_grad.iterations} iterations with error {layer_grad.err}")
print(f"Output Z_nograd: {Z_nograd}")
print(f"Terminated after {layer_nograd.iterations} iterations with error {layer_nograd.err}")

sol_grad = torch.sqrt(Z_grad + X)
print(f"Solution grad: {sol_grad}")
sol_nograd = torch.sqrt(Z_nograd + X)
print(f"Solution nograd: {sol_nograd}")

#get gradient of Z with respect to X
grad_grad = torch.autograd.grad(Z_grad, X, torch.ones_like(Z_grad))
print(f"Gradient grad: {grad_grad}")
grad_nograd = torch.autograd.grad(Z_nograd, X, torch.ones_like(Z_nograd))
print(f"Gradient nograd: {grad_nograd}")
print("Difference in gradients: ", grad_grad[0] - grad_nograd[0])

d = 1 / (1 - 1/(0.5 * torch.sqrt(Z_nograd + X)))
print(d)
