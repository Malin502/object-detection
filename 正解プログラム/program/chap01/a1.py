#!/usr/bin/python
# -*- coding: sjis -*-

import torch

def f(x1,x2,x3):
    return (x1 -2 * x2 -1)**2 + (x2 * x3 -1)**2 + 1

def f_grad(x1,x2,x3):
    z = f(x1, x2, x3)
    z.backward()
    return (x1.grad, x2.grad, x3.grad)

x1 = torch.tensor([1.], requires_grad=True) # x1 初期値
x2 = torch.tensor([2.], requires_grad=True) # x2 初期値
x3 = torch.tensor([3.], requires_grad=True) # x3 初期値
for i in range(50):
    g1, g2, g3 = f_grad(x1,x2,x3)
    x1 = x1 - 0.1 * g1 
    x2 = x2 - 0.1 * g2 
    x3 = x3 - 0.1 * g3 
    x1 = x1.detach().requires_grad_(True) 
    x2 = x2.detach().requires_grad_(True)
    x3 = x3.detach().requires_grad_(True)    
    print("x = [",x1.item(),x2.item(),x3.item(),"], f = ",
               f(x1,x2,x3).item())
