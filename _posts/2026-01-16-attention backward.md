---
layout: post
title: "Deriving the FlashAttention Backward Pass"
date: 2026-01-16 09:00:00
description: This supplementary note provides a complete mathematical derivation of the FlashAttention backward pass.
tags: dl, MLSystems
# categories: cs336
images:
  compare: true
  slider: true
---

The goal of this post is to supplement my [main post](http://timashov.ai/blog/2025/flash-attention) on Flash Attention 2 and to justify the identities and reformulations used in the optimized kernel, in particular the use of recomputation instead of storing intermediate attention matrices.

This document focuses exclusively on mathematical correctness and omits GPU implementation details, performance considerations, and Triton-specific code, which are covered in the main article.

---

## 1. Online softmax
Let $$x \in \mathbb{R}^V$$. 

The softmax is defined as $$y_i = \frac{\exp(x_i)}{\sum_{j=1}^V \exp(x_j)}$$.

A **numerically stable** implementation subtracts the maximum:
$$y_i = \frac{\exp(x_i - m)}{\sum_{j=1}^V \exp(x_j - m)}, \ m = \max_j x_j$$.

A naive computation requires **three passes** over x:
1. compute $$m$$
2. compute $$\sum_j \exp(x_j - m)$$
3. compute each $$y_i$$

To reduce this to **two passes**, Milakov and Gimelshein propose [online softmax](https://arxiv.org/pdf/1805.02867), which maintains a running maximum and running sum of exponents.

Initialize:  
$$\quad m_0 = -\infty, \ d_0 = 0$$.

For $$i = 1, \dots, V$$: (first loop)
$$\quad m_i = \max(m_{i-1}, x_i)$$,  
$$\quad d_i = d_{i-1} \cdot \exp(m_{i-1} - m_i) + \exp(x_i - m_i)$$.

After processing all elements, the softmax is given by:  
$$y_i = \frac{\exp(x_i - m_V)}{d_V}$$ (second loop).

This formulation enables streaming and tiled computation of softmax **without materializing the full denominator**. It is the **key enabler for tiled attention**, as it allows softmax normalization to be accumulated incrementally over blocks.

---

## 2. Backpropagation Through Matrix Multiplication
Let $$\mathbf{X} \in \mathbb{R}^{m \times n}, \ \mathbf{Y} \in \mathbb{R}^{n \times k}$$ and $$\mathbf{O} = \mathbf{XY}$$. Define $$\mathbf{dO} := \mathbf{\frac{\partial \mathcal{L}}{\partial O}}, \ \mathbf{dX} := \mathbf{\frac{\partial \mathcal{L}}{\partial X}}, \ \mathbf{dY} := \mathbf{\frac{\partial \mathcal{L}}{\partial Y}}$$.

Given $$\mathbf{dO}$$, I want to find $$\mathbf{dX}$$ and $$\mathbf{dY}$$.

$$
\mathbf{X} = \begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,n} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,n} \\
\cdots & \cdots & \cdots & \cdots \\
x_{m,1} & x_{m,2} & \cdots & x_{m,n} \\
\end{bmatrix}

\quad

\mathbf{Y} = \begin{bmatrix}
y_{1,1} & y_{1,2} & \cdots & y_{1,k} \\
y_{2,1} & y_{2,2} & \cdots & y_{2,k} \\
\cdots & \cdots & \cdots & \cdots \\
y_{n,1} & y_{n,2} & \cdots & y_{n,k} \\
\end{bmatrix}

\Rightarrow

\mathbf{O} = \mathbf{XY} = \begin{bmatrix}
\sum_{i=1}^n x_{1,i} \cdot y_{i,1} & \sum_{i=1}^n x_{1,i} \cdot y_{i,2} & \cdots & \sum_{i=1}^n x_{1,i} \cdot y_{i,k} \\
\sum_{i=1}^n x_{2,i} \cdot y_{i,1} & \sum_{i=1}^n x_{2,i} \cdot y_{i,2} & \cdots & \sum_{i=1}^n x_{2,i} \cdot y_{i,k} \\
\cdots & \cdots & \cdots & \cdots \\
\sum_{i=1}^n x_{m,i} \cdot y_{i,1} & \sum_{i=1}^n x_{m,i} \cdot y_{i,2} & \cdots & \sum_{i=1}^n x_{m,i} \cdot y_{i,k}
\end{bmatrix}
$$

Elementwise, $$o_{i,j} = \sum_{t=1}^n x_{i,t} \cdot y_{t,j}$$.

To compute gradients, I use **the chain rule** - $$\frac{\partial \mathcal{L}}{\partial x_{i,j}} = \sum_{\hat{i},\hat{j}} \frac{\partial \mathcal{L}}{\partial o_{\hat{i},\hat{j}}} \frac{\partial o_{\hat{i},\hat{j}}}{\partial x_{i,j}}$$.

The **key insight** is that the sum includes many terms that are zero $$o_{\hat{i},\hat{j}}$$ - only depends on $$x_{i,j}$$ when $$\hat{i}=i$$. More formally:

$$
\frac{\partial o_{\hat{i},\hat{j}}}{\partial x_{i,j}} = 
\begin{cases}
y_{j,\hat{j}} & \text{if } \hat{i} = i \\
0  & \text{if } \hat{i} \neq i
\end{cases}
\quad
\Rightarrow
\quad

\frac{\partial \mathcal{L}}{\partial x_{ij}} = \sum_{t=1}^{k} \frac{\partial \mathcal{L}}{\partial o_{i,t}} \cdot y_{j,t}
$$ 

Therefore, in matrix form: $$\mathbf{dX} = \mathbf{dO \ Y^T}$$.

Similarly, 

$$
\frac{\partial o_{\hat{i},\hat{j}}}{\partial y_{i,j}} = 
\begin{cases}
x_{\hat{i},i} & \text{if } \hat{j} = j \\
0  & \text{if } \hat{j} \neq j
\end{cases}
\quad
\Rightarrow
\quad

\frac{\partial \mathcal{L}}{\partial y_{ij}} = \sum_{t=1}^{m} \frac{\partial \mathcal{L}}{\partial o_{t,j}} \cdot x_{t,i}
$$ 

Therefore, in matrix form: $$\mathbf{dY} = \mathbf{X^T \ dO}$$.

---

## 3. Backpropagation Through Row-wise Softmax
Let $$\mathbf{S} \in \mathbb{R}^{m \times n}$$ and $$\mathbf{P} = \text{softmax}_\text{row}(\mathbf{S})$$. Similarly, define $$\mathbf{dP} := \frac{\partial \mathcal{L}}{\partial \mathbf{P}}, \ \mathbf{dS} := \frac{\partial \mathcal{L}}{\partial \mathbf{S}}$$.

Given $$\mathbf{dP}$$, I want to find $$\mathbf{dS}$$.

$$
\mathbf{P} = \text{softmax}_\text{row}(\mathbf{S}) = 
\begin{bmatrix}
\mathbf{\frac{exp(s_{1,1})}{\sum_{i=1}^{n} exp(s_{1,i})}} & \mathbf{\frac{exp(s_{1,2})}{\sum_{i=1}^{n} exp(s_{1,i})}} & \cdots & \mathbf{\frac{exp(s_{1,n})}{\sum_{i=1}^{n} exp(s_{1,i})}} \\
\mathbf{\frac{exp(s_{2,1})}{\sum_{i=1}^{n} exp(s_{2,i})}} & \mathbf{\frac{exp(s_{2,2})}{\sum_{i=1}^{n} exp(s_{2,i})}} & \cdots & \mathbf{\frac{exp(s_{2,n})}{\sum_{i=1}^{n} exp(s_{2,i})}} \\
\cdots & \cdots & \cdots & \cdots \\
\mathbf{\frac{exp(s_{m,1})}{\sum_{i=1}^{n} exp(s_{m,i})}} & \mathbf{\frac{exp(s_{m,2})}{\sum_{i=1}^{n} exp(s_{m,i})}} & \cdots & \mathbf{\frac{exp(s_{m,n})}{\sum_{i=1}^{n} exp(s_{m,i})}}
\end{bmatrix}
$$

Since softmax is applied **independently to each row**, $$\mathbf{p_{k,j}}$$ depends on $$\mathbf{s_{k,r}}$$ only within the same row $$\mathbf{k}$$:

 $$\mathbf{\frac{\partial \mathcal{L}}{\partial s_{k,r}}} = \sum_{j = 1}^{n} \mathbf{\frac{\partial \mathcal{L}}{\partial p_{k, j}}} \cdot \mathbf{\frac{\partial p_{k,j}}{\partial s_{k,r}}}$$

Fix a row $$\mathbf{k}$$ and write 

$$\mathbf{p_{k,j}=\frac{\exp(s_{k,j})}{\sum_{t=1}^n \exp(s_{k,t})}}$$

$$
\mathbf{\frac{\partial p_{k,j}}{\partial s_{k,r}}}= 
\begin{cases}
\mathbf{\frac{exp(s_{k,r}) \cdot \sum_{t=1}^{n}exp(s_{k,t}) - exp(s_{k,r}) \cdot exp(s_{k,r})}{(\sum_{t=1}^{n}exp(s_{k,t}))^2} = \textcolor{blue}{- p_{k,r} \cdot p_{k,r}} + p_{k,r}} & \text{if } j = r \\

\mathbf{\frac{-exp(s_{k,j}) \cdot exp(s_{k,r})}{(\sum_{t=1}^{n}exp(s_{k,t}))^2}= \textcolor{blue}{- p_{k,j} \cdot p_{k,r}}}  & \text{if } j \neq r
\end{cases}
$$ 

Next, 

$$\mathbf{\frac{\partial \mathcal{L}}{\partial s_{k,r}}} = \mathbf{\frac{\partial \mathcal{L}}{\partial p_{k, r}}} \cdot \mathbf{p_{k,r}} -\mathbf{\sum_{j=1}^n \frac{\partial \mathcal{L}}{\partial p_{k, j}}} \cdot (\mathbf{p_{k,j} \cdot p_{k,r}}) = \mathbf{p_{k,r}} \cdot (\mathbf{\frac{\partial \mathcal{L}}{\partial p_{k, r}}} - \mathbf{\sum_{j=1}^n \frac{\partial \mathcal{L}}{\partial p_{k, j}}} \cdot \mathbf{p_{k,j}})$$

Vectorizing over $$\mathbf{r}$$ yields the standard row-wise form: $$\mathbf{dS = P \odot (dP - rowsum(P \odot dP))}$$. 

Equivalently, for each row $i$, the same result can be written using the softmax Jacobian
$$
\mathbf{J}_i = \mathrm{diag}(\mathbf{P}_{i,}) - \mathbf{P}_{i,}^T\mathbf{P}_{i,},
$$
so that
$$
\mathbf{dS}_{i,} = \mathbf{dP}_{i,}\mathbf{J}_i = \mathbf{dP_{i,}} (\mathrm{diag}(\mathbf{P}_{i,}) - \mathbf{P}_{i,}^T\mathbf{P}_{i,})
$$.

---

## 4. Identity Eliminating Materialized Attention Weights
The following identity removes the need to store the attention weights matrix $$\mathbf{P}$$, enabling a single-pass backward computation.

Let $\mathbf{P}=\mathrm{softmax}_\mathrm{row}(\mathbf{S}), \quad \mathbf{O}=\mathbf{P}\mathbf{V}$, and $$\operatorname{diag}(\cdot)$$ extracts the diagonal as a vector. Then

$$
\textcolor{blue}{\mathrm{rowsum}(\mathbf{P}\odot\mathbf{dP})}
=
\operatorname{diag}\!\left(\mathbf{dP} \cdot \mathbf{P}^T\right)
=
\operatorname{diag}\!\left(\mathbf{dO} \cdot \mathbf{V}^T\mathbf{P}^T\right)
=
\operatorname{diag}\!\left(\mathbf{dO} \cdot (\mathbf{P}\mathbf{V})^T\right)
=
\operatorname{diag}\!\left(\mathbf{dO} \cdot \mathbf{O}^T\right)
=
\textcolor{blue}{\mathrm{rowsum}(\mathbf{O}\odot\mathbf{dO})}
$$

Defining
$$
\mathbf{D} :=\mathrm{rowsum}(\mathbf{O}\odot\mathbf{dO}) =\mathrm{rowsum}(\mathbf{P}\odot\mathbf{dP}),
$$
the softmax gradient can be written as
$$
\mathbf{dS} = \mathbf{P}\odot(\mathbf{dP} - \mathbf{D}),
$$
where $\mathbf{D}$ is broadcast across columns.
