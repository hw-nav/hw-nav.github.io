---
title: 'Global Convergence of a Stochastic Levenberg–Marquardt Algorithm Based on Trust Region'
date: 2024-10-26
permalink: /posts/2024/26/CarathéodoryTheorem/
tags:
  - Optimization
  - Stochastic Levenberg–Marquardt
---


## Paper

The final theorem is to prove

$$\liminf_{k\rightarrow \infty} \|\nabla R(W_k)\| = 0.$$



## Algorithm

I will push it in github. Main part of it is

```python
class StochasticLevenbergMarquardt_TrustRegion:
    def __init__(self, 
                 param     : np.ndarray  = None,        # param: 1-dim param 
                 pred_fn                 = None,        # Function: (w,x) -> Vector: size_y * 1
                 Jacobi_fn               = None,        # Function: (w,x) -> Matrix: size_y * size_param
                 filename  : str         = None, 
                 load_fn                 = None,        # Function: filename -> List[[x,y]]: batch_size * 2
                 lambda_LM : float       = 1e-3,
                 lambda_min: float       = 1e-9, 
                 p0        : float       = 1e-4, 
                 gamma     : float       = 4.0,
                 batch_size: int         = 1,           # default : 1
                 MAXITER   : int         = 10,
                 random_seed : int       = 42
                 ):
        self.param = param
        self.size_param = self.param.size
        self.lambda_LM = lambda_LM
        self.lambda_min = lambda_min
        self.p0 = p0
        self.gamma = gamma
        self.filename = filename
        self.batch_size = batch_size
        self.MAXITER = MAXITER
        self.pred_fn = pred_fn
        self.Jacobi_fn = Jacobi_fn
        self.load_fn = load_fn
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(self.random_seed)
            
        self.f = 0
        self.g = np.zeros((self.size_param, 1))                 # Vector: size_param * 1 , zeros_like(param)
        self.H = np.zeros((self.size_param, self.size_param))   # Matrix: size_param * size_param



    # zero -> f,g,H
    def grad_zero(self):
        self.f = 0
        self.g.fill(0)
        self.H.fill(0)

    # get -> f,g,H
    def backward(self, data):
        size_y = data[0][1].size
        for x,y in data:
            resi = (self.pred_fn(self.param, x) - y).reshape((size_y,1))  # Vector: size_y * 1 
            grad = self.Jacobi_fn(self.param,x)                           # Matrix: size_y * size_param 
            self.f   += np.sum(np.square(resi))
            self.g   += grad.T @ resi
            self.H   += grad.T @ grad
        self.f *= 0.5/self.batch_size
        self.g /= self.batch_size
        self.H /= self.batch_size  

    def step(self):
        eps = 1e-10
        k = 0
        # 循环
        while k < self.MAXITER:
            data = self.load_fn(self.filename, batch_size=self.batch_size, random_seed=self.random_seed)
            self.grad_zero()
            self.backward(data)

            g_norm = np.linalg.norm(self.g)
            if g_norm < eps:
                print("Stop By Norm, Epoch = {}, Norm(g) = {}".format(k, g_norm))
                break

            d = -np.linalg.solve(self.H + self.lambda_LM * g_norm * np.eye(6,6), self.g)
            wd = self.param+d.flatten()
            fd = 0.5 * np.mean([np.sum(np.square(self.pred_fn(wd,x) - y)) for x,y in data])

            r = (fd-self.f)/(self.g.T @ d + 0.5 * d.T @ self.H @ d)
            if r >= self.p0:
                self.param = wd
                lambda_LM = max(self.lambda_LM/self.gamma, self.lambda_min)
            else:
                lambda_LM *= self.gamma
            k += 1

            if k % (self.MAXITER // 10) == 0:
                print("{}: Loss = {}".format(k, self.f))
                print("param - w* = {}".format(self.param-w_opt))
        if k == self.MAXITER:
            print("Stop By MAXITER, Epoch = {}, Norm(g) = {}".format(k, g_norm))
```

I want to make a class contain all the algorithm, maybe it can be simple to understand.






## Reference

[1] Shao W Y, Fan J Y. Global Convergence of a Stochastic Levenberg–Marquardt Algorithm Based on Trust Region[J]. Journal of the Operations Research Society of China, 2024: 1-23.