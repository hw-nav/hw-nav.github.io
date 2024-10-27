---
title: 'On The Complexity of A Stochastic Levenberg-Marquardt Method'
date: 2024-10-27
permalink: /posts/2024/10/On-The-Complexity-of-A-Stochastic-LM-Method/
tags:
  - Optimization
  - Stochastic Levenberg–Marquardt
---

Similar to [Global Convergence of a Stochastic Levenberg–Marquardt Algorithm Based on Trust Region](https://hw-nav.github.io/posts/2024/10/Global-Convergence-of-a-SLM-Algorithm-Based-on-Trust-Region/)

Main result: Define

$$N_\varepsilon = \inf \{k\in \mathbb{N} \mid \|\nabla R(W_k)\| \leq \varepsilon\}$$

then

$$E(N_\varepsilon) \leq \zeta_6 \varepsilon^{-2} + \frac{1}{1-\exp(-\zeta_7)}.$$




## Reference

[1] Shao W, Fan J. On the complexity of a stochastic Levenberg-Marquardt method[J]. Journal of Industrial and Management Optimization, 2024, 20(3): 1011-1027.