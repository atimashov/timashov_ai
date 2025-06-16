---
layout: post
title: "Gradient-Based Optimization: Theory, Practice, and Evolution"
date: 2025-07-26 09:00:00
description: sgd, momentum, adagrad, adam, and beyond
tags: dl-basics, lecture-1
categories: cs336
# thumbnail: assets/img/9.jpg
images:
  compare: true
  slider: true
---

In this post, I’ll break down the core ideas behind popular optimizers used in Deep Learning. Starting with Vanilla Gradient descent, I’ll explore how momentum, adaptive learning rates, and their combinations (like Adam) help models converge faster and more reliably — and the intuition behind each step. Finally, I'll mention briefly latest developments.

## Vanilla Gradient Descent

**Core Idea:** Move weights in the direction of the negative gradient to minimize the loss.

In Deep Learning, the typical goal is to minimize a **loss function** - we may also call it cost function - **over a training dataset**.  
**Gradient Descent** is the foundation of most Deep Learning optimizers. In practice, we compute the gradient of loss $$f(W)$$ with respect to the model’s weights $$W$$ and update iteratively the weights accordingly. On each step, we subtract the gradient of the loss with respect to the weights, scaled by a learning rate:

$$W_{k+1} = W_k - \text{lr} * \nabla f(W_k)$$

In Python code, this looks like:
```python
while cond:
    weights_grad = evaluate_gradient(loss_func, data, weights)
    weights -= learning_rate * weights_grad    
```

Each iteration updates weights in the direction that locally reduces the loss. The stopping condition may be a fixed number of steps or a threshold on loss improvement. For instance, in LLMs it is generally iteration over one epoch - over all data one time.

Let’s walk through a **concrete example**. Suppose: $$W = \begin{bmatrix} w_1\\ w_2 \end{bmatrix}$$ is a $$\text{2D}$$ vector, and $$f(W) = w_1^2 + bw_2^2 = W^T \begin{bmatrix} 1 & 0\\ 0 & b \end{bmatrix} W$$.

Then:

1. Minimum: $$\min f(W) = 0$$ at $$W = \begin{bmatrix} 0\\ 0 \end{bmatrix}$$ ($$\text{argmin}$$).
2. Gradient: $$\nabla_W f(W) = \begin{bmatrix} 2w_1\\ 2bw_2 \end{bmatrix}$$.
3. Update rule: $$W_{k+1} =W_k - \text{lr} * \nabla f(W_k) = \begin{bmatrix} w_1\\ w_2 \end{bmatrix} - \text{lr} \begin{bmatrix} 2w_1\\ 2bw_2 \end{bmatrix} = \begin{bmatrix} (1-2*\text{lr})w_1\\ (1-2*b*\text{lr})w_2 \end{bmatrix}$$.

I simulate gradient descent from initial point $$W^0 = \begin{bmatrix} 0.5 \\ 0.5\end{bmatrix}$$ with learning rate $$\text{lr} = 0.3$$ in two cases (different values of $$b$$):

<img src="/assets/img/optimizers/five_steps.png" alt="Img.1: Five steps of gradient descent" style="width:100%;">

**Observation:**
- Progress along different dimension can vary dramatically if conditioning is poor.
- The path may "zig-zag" - commonly when the loss landscape is elongated.

### Stochastic Gradient Descent
Vanilla Gradient Descent assumes we can compute gradients **over the entire dataset** at every step. For large datasets, this becomes computationally infeasible.

A more practical alternative is Stochastic Gradient Descent (SGD): Instead of using all data, SGD approximates the gradient using a **random mini-batch** at each step. This introduces some noise but dramatically improves efficiency and scalability.

```python
while cond:
    batch = sample_batch(data)
    weights_grad = evaluate_gradient(loss_func, batch, weights)
    weights -= learning_rate * weights_grad
```
Today, almost every optimizer — from SGD+Momentum to Adam, and beyond — is built on this stochastic mini-batch principle.

**Challenges with SGD:**
1. Noisy behaviour  
Gradients computed on a mini-batch fluctuate due to sample variance. It is not always bad in fact - for example it can be helpful in avoiding sharp local minima.

2. Poor conditioning  
Loss surface has directions with very different curvature - some steeps, and other flat. Formally, it is a large ratio between the largest and smallest eigenvalues of the Hessian. As a result, we observe "zig-zag" behaviour and slow convergence in flatter directions.

<img src="/assets/img/optimizers/poor_conditioning.png" alt="Img.2: Poor conditioning" style="width:100%;">

3. Local minimas and saddle points.  
If the gradient vanishes, SGD may stall — even if it’s not a true minimum.

<img src="/assets/img/optimizers/local_minima.png" alt="Img.3: Local minima and saddle points" style="width:100%;">

---

## Momentum and Acceleration
To improve SGD’s performance — particularly its zig-zagging in poorly conditioned loss landscapes and stucking due to saddle points and local minimas — we can add momentum, which smooths updates over time by accumulating gradients.

### SGD+Momentum
**Core idea:**
Instead of moving directly in the direction of the current gradient, we use a running sum of gradients (like velocity in physics) to smooth updates. Often scale $$\rho$$ is aplied.  
$$v_{t+1} = \rho v_t + \nabla f(w_t)$$  
$$w_{t+1} = w_t - \alpha v_{t+1}$$

Python-style pseudocode:
```python
velocity = 0
while cond is True:
    batch = sample_batch(data)
    weights_grad = evaluate_gradient(loss_func, batch, weights)
    velocity = rho * velocity + weights_grad
    weights -= learning_rate * velocity
```
Tipically $\text{rho}$ is between $0.9$ and $0.99$.

**Benefits:**
1. Reduces zig-zagging in poorly conditioned settings.
2. Helps escape local minima and saddle points using accumulated velocity.
3. Averages out noise from stochastic gradients.

### Nesterov Momentum
**Core idea:**
Instead of applying the gradient at the current location, we look ahead in the direction of momentum.  
$$v_{t+1} = \rho v_t - \alpha \nabla f(w_t + \rho v_t)$$  
$$w_{t+1} = w_t + v_{t+1}$$

For consistency with classical momentum, we can replace $$\hat{v} = -v \textcolor{red}{\text{(probably add "hat")}}$$:  
$$v_{t+1} = \rho v_t + \alpha \nabla f(w_t + \rho v_t)$$  
$$w_{t+1} = w_t - v_{t+1}$$

To keep working with gradients in a current point, we make one more variables' change: $$\hat{w_t} = w_t+\rho v_t$$:

$$v_{t+1} = \rho v_t + \alpha \nabla f(\hat{w_t})$$  
$$\hat{w_{t+1}} = \textcolor{blue}{w_{t+1} +\rho v_{t+1} = w_t -v_{t+1} + \rho v_{t+1} + \rho v_t - \rho v_t} = \hat{w_t} - \rho v_t - (1-\rho)v_{t+1}$$

Python-style pseudocode:
```python
velocity = 0
while cond:
    batch = sample_batch(data)
    weights_grad = evaluate_gradient(loss_func, batch, weights)
    old_velocity = velocity
    velocity = rho * velocity + learning_rate * weights_grad
    weights -= rho * old_velocity + (1 - rho) * velocity
```
$$\textcolor{red}{\text{Check / Modify summary: }}$$

Nesterov gives a slight performance boost over vanilla momentum by being “a step ahead” — especially in convex problems or when updates start to overshoot.

---

## Accumulating Squared Gradients: AdaGrad and RMSProp
Gradient-based optimization can suffer from inconsistent step sizes across dimensions — especially under poor conditioning.  
A powerful solution is to **adapt the step size** per parameter, using **accumulated squared gradients** to normalize the updates.

### AdaGrad
**Core idea:** Scale down learning rates for parameters that receive large gradients over time.
Proposed by John Duchi et al., AdaGrad accumulates the **sum of squared gradients** and divides each gradient update by the square root of this sum.

$$\text{sq_grad}_{t+1} = \text{sq_grad}_t + (\nabla f(x))^2$$  
$$w_{t+1} = w_t - \alpha \frac{\nabla f(x)}{\text{sq_grad}_{t+1} + \epsilon}$$ 

**Python-style pseudocode:**
```python
grad_squared_sum = 0
while cond:
    batch = sample_batch(data)
    weights_grad = evaluate_gradient(loss_func, batch, weights)
    grad_squared_sum += weights_grad ** 2
    weights -= learning_rate * weights_grad / (torch.sqrt(grad_squared_sum) + 1e-7)
```
It naturally adjusts learning rate based on historical gradient magnitude, and helps to balance poor conditioning. However, there is a **problem with Adagrad**: step size eventually shrinks to nearly zero due to the growing sum of squared gradients over time.

### RMSProp
**Core idea:** Fix AdaGrad’s vanishing steps problem by applying a decay factor to the accumulated sum.  
$$\textcolor{red}{\text{Check:}}$$ Originally introduced by Geoff Hinton in his lectures, RMSProp solce the problem of diminishing steps in optimization.

**Python-style pseudocode:**
```python
grad_squared_sum = 0
while cond:
    batch = sample_batch(data)
    weights_grad = evaluate_gradient(loss_func, batch, weights)
    grad_squared_sum = decay_rate * grad_squared_sum + (1 - decay_rate) * weights_grad ** 2
    weights -= learning_rate * weights_grad / (torch.sqrt(grad_squared_sum) + 1e-7)
```
Tipically, $$\text{decay_rate}$$ is generally $$0.9$$ or $$0.99$$. 

---

## Adam
**Core idea:** Combine the benefits of momentum and adaptive learning rates into a single optimizer.  
Adam (short for *Adaptive Moment Estimation*) maintains both:
- An exponentially decaying average of past gradients (**first moment**),
- And an exponentially decaying average of past squared gradients (**second moment**).

$$\textcolor{red}{\text{Check if decaying is appropriate here}}$$

The update rule looks like this:
```python
# Naive Version
first_moment = 0
second_moment = 0
while cond:
    batch = sample_batch(data)
    weights_grad = evaluate_gradient(loss_func, batch, weights)
    first_moment = beta1 * first_moment + (1 - beta1) * weights_grad
    second_moment = beta2 * second_moment + (1 - beta2) * weights_grad ** 2
    x -= learning_rate * first_moment / (torch.sqrt(second_moment) + 1e-7)
```
On early steps (especially the first few), both `first_moment` and `second_moment` are close to zero.
This leads to **biased estimates** and can result in **overly large updates**, not because of gradient shape but due to initialization.

To fix this, Adam applies bias correction factors:
```python
first_moment = 0
second_moment = 0
for t in range(1, num_iterations + 1):
    weights_grad = evaluate_gradient(loss_func, data, weights)
    first_moment = beta1 * first_moment + (1 - beta1) * weights_grad
    second_moment = beta2 * second_moment + (1 - beta2) * weights_grad ** 2
    first_unbias = first_moment / (1 - beta1 ** t)
    second_unbias = second_moment / (1 - beta2 ** t)
    x -= learning_rate * first_unbias / torch.sqrt(second_unbias) + 1e-7
```
Adam is a **default choice** for many problems. Proposed parameters to start with:
```python
beta1 = 0.9
beta2 = 0.99
learning_rate = 1e-3 # or 5e-4
```
Adam combines the stability of Momentum with adaptive step size from RMSProp, bias correction improves performance early in training. At the same time, it can sometimes $$\textcolor{red}{\text{overfit or oscillate}}$$ if used without regularization

### AdamW
**Core Idea:** Decouple weight decay from the gradient-based optimization step.

To understand what is the misconception with Adam (and other adaptative gradient algorithms), we need to recall what is weight decay, $$L_2$$ regularization, and why it is not the same as.

To understand the need for AdamW, we need to revisit two related but different concepts: **weight decay** and **$$L_2$$ regularization**. They are often treated as equivalent — but that only holds for standard SGD. In adaptive optimizers, they **behave differently**.

**Weight decay**  
Weight decay was introduced as an explicit update rule that shrinks the weights at each step ([Hanson & Pratt (1988)](https://proceedings.neurips.cc/paper/1988/file/1c9ac0159c94d8d0cbedc973445af2da-Paper.pdf)):

$$\theta_{t+1} = \textcolor{red}{(1-\lambda)}\theta_t - \alpha \nabla f_t(\theta_t) = \theta_t - \alpha \nabla f_t(\theta_t) - \textcolor{red}{\lambda}\theta_t$$

This directly subtracts a portion of the weights themselves on every step — independent of the loss gradient.

**L2 regularization**  
$$L_2$$ regularization is applied by **modifying the loss function** to penalize large weights:

$$f(\theta)_{L_2} = f(\theta) + \lambda \sum \theta^2$$

When we take the gradient of this new loss, we get:

$$\nabla f_{L_2}(\theta) = \nabla f(\theta) + \textcolor{red}{2\lambda \theta}$$

At first glance, this looks similar to weight decay. And for SGD, it is — they’re equivalent up to scale. But this **equivalence breaks** down in adaptive optimizers like Adam.

**Why This Breaks in Adam**  
In Adam, gradients are scaled by second-moment estimates (running average of squared gradients). If we add the **$$L_2$$ term** into the gradient, it becomes part of the **adaptive scaling**. This causes distortions:

$$g_{t+1} \leftarrow \nabla f(\theta_t) + \textcolor{red}{2\lambda \theta_t}$$

Then:

$$g_{t+1}^2 \leftarrow \nabla f(\theta_t)^2 + 4\lambda \nabla f(\theta_t) \theta_t + 4\lambda^2 \theta_t^2$$

This means that the regularization term entangles with the loss gradient and gets adaptively rescaled — which is not what we want for weight decay.

**AdamW Fix**  
The **AdamW** paper **proposes** to remove the L2 term entirely from the loss and instead apply weight decay **outside the gradient calculation**:
1. No L2 term in the loss. Gradients come only from the task loss.
2. Only apply weight decay in the update step:  
$$w_{t+1} = w_t - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + 2\textcolor{green}{\lambda w_t} \right)$$

This subtle change restores the correct role of weight decay — shrinking weights uniformly, regardless of the optimizer’s adaptive scaling. As a result, **AdamW** became **the default choice** for training large models across NLP and vision.

---

## Recent Advances


---

## Practical Tricks

---

## FLOPs and Optimizer Efficiency

---

## Conclusion
- first idea
- second idea
- third idea

Questions I have:
- What if we make learing rate as a parameter and use it to optimize our loss function?