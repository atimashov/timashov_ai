---
layout: post
title: "Backpropagation: From Intuition to FLOPs"
date: 2025-04-28 09:00:00
description: gradient calculation, flow and FLOPs
tags: dl-basics, lecture-1
categories: cs336
# thumbnail: assets/img/9.jpg
images:
  compare: true
  slider: true
---

This post dives into what happens during the backward pass of a neural network.
We’ll start with a simple scalar example and build up to full matrix-based gradient derivations — showing not only how gradients are computed, but also how to count **FLOPs** along the way.

## Introduction
In the previous [post](timashov.ai/blog/2025/compute/) I covered how to construct tensors and pass them through operations - this is known as the "forward pass".

The next step in deep learning is computing **gradients** and making an optimizer step - the **backward pass**.

Let's look at a simple example: $y=0.5 \times (x \times w-5)^2$.  
**Forward pass:** compute loss
```
x = torch.Tensor([1.0, 2.0, 3.0]) # we don't need gradients
w = torch.Tensor([1.0, 1.0, 1.0], requres_grad = True) # we need gradients
pred_y = x @ w
loss  = 0.5 * (pred_y - 5).pow(2)
```

**Backward pass:** compute gradients
```
loss.backward()
assert loss.grad is None
assert pred_y.grad is None
assert x.grad is None
assert torch.equal(w.grad, torch.tensor(1.0, 2.0, 3.0))
```

---

## Matrix Gradients by Hand and Backward FLOPs
Let's consider a more realistic setup. We have
- input matrix ***X*** with dimensions $B \times D$
- hidden layer(1) ***W1*** with dimensions $D \times D$
- hidden layer(2) ***W2*** with dimensions $D \times C$ (outputs ***C*** classes)

The model is 
- $H_1 = XW_1$ (the dimensions: $$(B \times D) \ @ \ (D \times D) \to B \times D$$)
- $H_2 = H_1W_2$ (the dimensions: $$(B \times D) \ @ \ (D \times C) \to B \times C$$)
- $\text{loss} = \text{mean}(H_2^2)$

To perform a **backward step**, we will need **gradients with respect to all weights**: $\frac{\text{dloss}}{\text{dw}_1}$ and $\frac{\text{dloss}}{\text{dw}_2}$.

Let's visualise matrices, this helps to build intuition for how gradients are calculated step by step.

$$
X = 
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1D} \\
x_{21} & x_{22} & \cdots & x_{2D} \\
\cdots & \cdots & \cdots & \cdots \\
x_{B1} & x_{B2} & \cdots & x_{BD} \\
\end{bmatrix}
\qquad
W_1 = 
\begin{bmatrix}
w^{(1)}_{11} & w^{(1)}_{12} & \cdots & w^{(1)}_{1D} \\
w^{(1)}_{21} & w^{(1)}_{22} & \cdots & w^{(1)}_{2D} \\
\cdots & \cdots & \cdots & \cdots \\
w^{(1)}_{D1} & w^{(1)}_{D2} & \cdots & w^{(1)}_{DD} \\
\end{bmatrix}
\qquad
W_2 = 
\begin{bmatrix}
w^{(2)}_{11} & w^{(2)}_{12} & \cdots & w^{(2)}_{1C} \\
w^{(2)}_{21} & w^{(2)}_{22} & \cdots & w^{(2)}_{2C} \\
\cdots & \cdots & \cdots & \cdots \\
w^{(2)}_{D1} & w^{(2)}_{D2} & \cdots & w^{(2)}_{DC} \\
\end{bmatrix}
$$

### Forward pass
We’ll walk through each layer’s output to build a concrete understanding of how activations are computed.

**Step1:** Compute ***H₁***

$$
H_1 = XW_1 = 
\begin{bmatrix}
\sum_{k=1}^D x_{1k}w^{(1)}_{k1} & \sum_{k=1}^D x_{1k}w^{(1)}_{k2} & \cdots & \sum_{k=1}^D x_{1k}w^{(1)}_{kD} \\
\sum_{k=1}^D x_{2k}w^{(1)}_{k1} & \sum_{k=1}^D x_{2k}w^{(1)}_{k2} & \cdots & \sum_{k=1}^D x_{2k}w^{(1)}_{kD} \\
\cdots & \cdots & \cdots & \cdots \\
\sum_{k=1}^D x_{Bk}w^{(1)}_{k1} & \sum_{k=1}^D x_{Bk}w^{(1)}_{k2} & \cdots & \sum_{k=1}^D x_{Bk}w^{(1)}_{kD}
\end{bmatrix}
$$

**Step2:** Compute ***H₂***

$$
H_2 = H_1W_2 = 
\begin{bmatrix}
\sum_{k=1}^D h^{(1)}_{1k}w^{(2)}_{k1} & \sum_{k=1}^D h^{(1)}_{1k}w^{(2)}_{k2} & \cdots & \sum_{k=1}^D h^{(1)}_{1k}w^{(2)}_{kC} \\
\sum_{k=1}^D h^{(1)}_{2k}w^{(2)}_{k1} & \sum_{k=1}^D h^{(1)}_{2k}w^{(2)}_{k2} & \cdots & \sum_{k=1}^D h^{(1)}_{2k}w^{(2)}_{kC} \\
\cdots & \cdots & \cdots & \cdots \\
\sum_{k=1}^D h^{(1)}_{Bk}w^{(2)}_{k1} & \sum_{k=1}^D h^{(1)}_{Bk}w^{(2)}_{k2} & \cdots & \sum_{k=1}^D h^{(1)}_{Bk}w^{(2)}_{kC}
\end{bmatrix}
$$

More generally, element $h^{(2)}_{ij} = \sum _{k=1}^D h^{(1)} _{ik} w^{(2)} _{kj}$.

**Step3:** Compute the final loss  
$$  
\text{loss} = \frac{1}{BC}\sum_{i=1}^B \sum_{j=1}^C (h^{(2)}_{ij})^2
$$

Forward FLOPs (excluding loss): `num_forward_flops = 2 * B * D * D + 2 * B * D * C`.

Now that we’ve computed the forward activations, it’s time to trace gradients backward using the chain rule.

### Backward pass
To compute gradients, we use the **chain rule** - "a formula that expresses the derivative of the composition of two differentiable functions" ([wiki](https://en.wikipedia.org/wiki/Chain_rule)). In Deep Learning it is often more intuitive to write it using Leibniz notation: if $z$ depends on $y$ and $y$ depends on $x$, then $z$ depends on $x$ and we can write it as

$$\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}$$


#### <u>Gradient with respect to H₂</u>
Now that we’ve defined the loss, the first step in the backward pass is to compute the gradient of the loss with respect to the model's output, ***H₂***:

$$\frac{\text{dloss}}{dh^{(2)}_{ij}}=\frac{1}{BC}\cdot 2 \cdot dh^{(2)}_{ij} \quad \text{(Only one tem of the sum depends on } h^{(2)}_{ij} \text{)}$$

Or in matrix form:

$$\frac{\text{dloss}}{dH_2}=\frac{2}{BC}H_2$$

#### <u>Gradient with respect to W₂</u>
With $$\frac{\text{dloss}}{dH_2}$$ computed, we can calculate the gradient w.r.t. the second layer weights, ***W₂***, using the chain rule.  
We start with:

$$\frac{\text{dloss}}{dw^{(2)}_{ij}} =  \sum_{i^{'}=1}^B \sum_{j^{'}=1}^C \frac{\text{dloss}}{dh^{(2)}_{i^{'}j^{'}}} \cdot \frac{dh^{(2)}_{i^{'}j^{'}}}{dw^{(2)}_{ij}}$$

The **key insight** is that the sum includes many terms that are zero - $$h^{(2)}_{i^{'}j^{'}}=\sum_{k=1}^{D} h^{(1)}_{i^{'}k}w^{(2)}_{kj^{'}}$$ only depends on $$w^{(2)}_{ij}$$ when $$j^{'}=j$$.  

Let’s derive it more formally:

$$
\frac{dh^{(2)}_{i^{'}j^{'}}}{dw^{(2)}_{ij}} = 
\begin{cases}
h_{i^{'}i} & \text{if } j^{'} = j \\
0  & \text{if } j^{'} \neq j
\end{cases}
\quad
\Rightarrow
\quad

\frac{\text{dloss}}{dw^{(2)}_{ij}} =  \sum_{k=1}^{B} \frac{\text{dloss}}{dh^{(2)}_{kj}} \cdot h^{(1)}_{ki} = \sum_{k=1}^{B} (h^{(1)})^T_{ik} \cdot \frac{\text{dloss}}{dh^{(2)}_{kj}}
$$ 

In matrix form it will look this way:

$$\frac{\text{dloss}}{dW_2} = H_1^T \frac{\text{dloss}}{dH_2} \text{ (dimensions: } D \times C\text{)}$$

This is a matrix multiplication between a $$D \times B$$ and a $$B \times C$$ matrix - resulting in a $$D \times C$$ output.  
The number of backward FLOPs so far: `num_backward_flops = 2 * B * D * C`.

#### <u>Gradient with respect to H₁</u>
Next, we need to backpropagate through ***W₂*** into the hidden activations ***H₁***.
This is needed to continue the chain toward ***W₁***. Following the same idea, we can get:

$$\frac{\text{dloss}}{dH_1} = \frac{\text{dloss}}{dH_2}W_2^T \text{ (dimensions: } B \times D\text{)}$$

The number of backward FLOPs so far: `num_backward_flops = 4 * B * D * C = 2 * B * D * C + 2 * B * D * C`.

#### <u>Gradient with respect to W₁</u>
Now we compute gradients for the first layer weights, $$W_1$$ - this completes the parameter gradient path for training.

$$\frac{\text{dloss}}{dw^{(1)}_{ij}} = \sum_{i^{'}=1}^B \sum_{j^{'}=1}^D \frac{\text{dloss}}{dh^{(1)}_{i^{'}j^{'}}} \cdot \frac{dh^{(1)}_{i^{'}j^{'}}}{dw^{(1)}_{ij}} = \sum_{k=1}^D \frac{\text{dloss}}{dh^{(1)}_{kj}} \cdot \frac{dh^{(1)}_{kj}}{dw^{(1)}_{ij}} = \sum_{k=1}^D \frac{\text{dloss}}{dh^{(1)}_{kj}} \cdot x_{ki}$$

So, in matrix form:

$$\frac{\text{dloss}}{dW_1} = \frac{\text{dloss}}{dH_1^T}X \text{ (the dimensions: } D \times D\text{)}$$

Updated backwards FLOPs: `num_backward_flops = 4 * B * D * C + 2 * B * D * D`.

#### <u>Gradient with respect to X</u>
We can also compute gradients w.r.t. the input $X$ — although this **isn’t needed during training**, it’s useful for things like input sensitivity or adversarial attacks. Following the same chain rule logic:

$$\frac{\text{dloss}}{dX} = \frac{\text{dloss}}{dH_1}W_1^T \text{ (the dimensions: } B \times D\text{)}$$

The number of backward FLOPs: `num_backward_flops = 4 * B * D * C + 4 * B * D * D`.
This is exactly $2\times$ **the number of FLOPs** compared to the forward pass.

---

## Conclusion
- **Understanding the math** behind helps in debugging, optimizing, and designing new architectures.
- The **backward pass** is built on the **chain rule**, propagated layer by layer.
- Backward pass requires **~2x the GLOPs** of the forward pass.
<!-- Understanding data types is the foundation of building efficient AI systems.  
It affects not only how we design models but also how fast and how large they can be.

In future posts, I’ll dive deeper into **resource accounting** — covering both **memory** and **FLOPS**.
For memory, I’ll go beyond inputs to include gradients, intermediate activations, and other internal components of deep learning models. -->
