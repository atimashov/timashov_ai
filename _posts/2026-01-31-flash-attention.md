---
layout: post
title: "FlashAttention-2 in Triton: From GPU Mental Models to Kernel Performance"
date: 2026-01-31 09:00:00
description: Implementing FlashAttention-2 in Triton, with a practical intro to GPUs and NVIDIA Nsight Systems.
tags: dl, MLSystems
categories: cs336
images:
  compare: true
  slider: true
---

<span style="color:red">[THIS IS DRAFT!!!]</span>

The goal of this post is simple: implement **FlashAttention-2** in **Triton**.

To do that in a meaningful way, we first need just enough intuition about **how GPUs execute ML workloads** and how to **profile them**. I’ll start with a practical overview of GPU internals and memory, then show how to use **NVIDIA Nsight Systems** to see where time actually goes. After that, I’ll dive into a Triton implementation of FlashAttention-2.

This is not a full “GPU architecture” post. If you want to go deeper, I highly recommend:

- Horace He’s posts: [here](https://horace.io/brrr_intro.html) and [here](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications)
- Aleksa Gordić’s [matmul blog](https://www.aleksagordic.com/blog/matmul)
- Stanford’s fantastic [CS336 lectures](https://stanford-cs336.github.io/spring2025/)

For profiling, I’ll only cover the parts of **Nsight Systems** that I actually use while building FlashAttention-2. For anything beyond that, the official [documentation](https://developer.nvidia.com/nsight-systems) is the best resource.

View the code related to this post on [GitHub](https://github.com/atimashov/llms-from-scratch).

---

## I. GPU Mental Model for FlashAttention

#### 1. Definitions and Logical Memory Model of GPU
**CPUs and GPUs** are designed for fundamentally **different goals** and therefore have
very different architectures. A CPU executes a relatively small number of threads,
each capable of handling complex control flow and sequential logic. In contrast, a
**GPU can execute thousands of threads in parallel**, but each thread performs a much
simpler task.

These design goals are reflected in the hardware. CPUs dedicate a large fraction of
silicon area to sophisticated control logic, while GPUs consist of many small,
simple execution units optimized for data-parallel workloads. A high-level mental
model of this difference is illustrated below.

<img src="/assets/img/triton/cpu_gpu.png" alt="Img.1: CPU vs GPU" style="width:100%;">

In short, CPUs are optimized for **latency**, while GPUs are optimized for
**throughput**. Individual GPU threads may run slower than CPU threads, but by
executing thousands of them concurrently, GPUs achieve much higher overall
throughput.

**Important definitions:**
* **Thread** – the smallest unit of execution. Threads execute the same instruction
  stream but operate on different data (see [Control Divergence](#control-divergence)).
* **Warp** – a fixed-size pack of **32 threads** that execute in lockstep on a SM.
  Control flow divergence within a warp leads to serialized execution.
* **Thread block** – a group of threads scheduled on the **same SM**, sharing access
  to on-chip shared memory.

**Execution model:**  
GPUs execute threads in warps across multiple Streaming Multiprocessors (SMs), following a SIMT (Single Instruction, Multiple Threads) execution model. Each SM contains many execution units (often referred to as CUDA cores) that collectively execute warps assigned to the SM.

At any given time, an SM issues instructions at the warp level: all threads within a warp execute the same instruction in lockstep, each operating on different data.

**Data Movement:**  
Before writing GPU kernels, it is essential to understand the GPU memory hierarchy:
* **L1 cache and shared memory (SRAM)** reside inside each SM, making them the fastest memories accessible by threads. While L1 cache is managed by hardware, shared memory is explicitly controlled by the programmer.
* **L2 cache** is shared across the GPU and sits outside individual SMs. Accessing L2 is slower than on-chip memory but significantly faster than global memory. Like L1 cache, it is managed by hardware, and not controlled by the programmer.
* **Global memory (DRAM)** is the main GPU memory visible to the programmer. It is
  located off-chip and has much higher latency than on-chip memory.

As models scale, additional levels of the hierarchy come into play: multiple GPUs within a server, servers within racks, and racks within clusters. Each new level dramatically increases data transfer cost, making **memory-aware design critical for performance**.

**Cuda Cores vs Tensor Cores:**
CUDA Cores are general-purpose scalar/vector execution units capable of performing
standard floating-point operations. **Tensor Cores** are specialized units designed
specifically for matrix-multiply-and-accumulate (MMA) operations, which dominate
deep learning workloads.

When operating on supported data types (e.g. FP16, BF16, INT8), **Tensor Cores can
deliver an order-of-magnitude higher throughput** than CUDA Cores. Fully utilizing
Tensor Cores requires careful kernel design, and data layout.

#### 2. Coalescing memory
**Idea:** Global memory (DRAM) is accessed in fixed-size chunks ("burst mode"). Each load request fetches an entire section, not individual bytes. While the exact data returned is not arbitrary, it is predictable based on the accessed addresses.

In the illustration below, I consider a 16-byte address range with a 4-byte memory transaction size. If I read values at addresses 0 through 7 sequentially (4 values at the time), the GPU performs two memory transactions: one for addresses 0–3 and one for addresses 4–7. All fetched data is used.

In contrast, if we access the same addresses in a strided pattern — (0, 4), (1, 5), (2, 6), (3, 7) — each access touches a different memory transaction. As a result, the same 8 bytes are fetched multiple times, leading to redundant memory traffic. In this case, 75% of the transferred data is unused, and the number of memory transactions is twice the required.

<img src="/assets/img/triton/burst_mode.png" alt="Img.2: Burst mode" style="width:100%;">

On modern GPUs, memory transactions are typically at least 128 bytes. To avoid redundant memory accesses, threads within a warp should access contiguous memory addresses so that their loads fall within the same memory transaction.

For a deeper dive on memory access patterns (including cool illustrations), I highly recommend the [blog post](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications) by Horace He.

<a id="tiling"></a>
#### 3. Tiling 
Tiling is a technique for **reducing global memory accesses** by grouping threads so that they load and reuse contiguous blocks of data, building on the idea of coalesced memory access. Once a tile is loaded into SRAM, as much computation as possible is performed before evicting it.

The rough algorythns is below:
1. Split matrices on a smaller "tiles"
2. Iterate over tiles of $$M$$;
    1. Load tile $$M_i$$ into shared memory
    2. Iterate over tiles of $$N$$
        1. Load tile $$N_j$$ into shared memory
        2. Compute partial sums of $P_{i,j}$
        3. Unload tile $$N_j$$
    3. Unload tile tile $$M_i$$

The illustration of this approach is shown below.

<div style="text-align: center;">
  <img src="/assets/img/triton/tiling.png"
       alt="Img.3: Tiling example"
       style="width:60%;">
</div>

The main benefit of tiling is that each element loaded from global memory is used the maximum amount of times in arithmetic operations, reducing the number of DRAM accesses.

However, tiling introduces a few practical challenges that can reduce utilization
and hurt performance:
1. Tile sizes may not divide matrix dimensions, and the total number of tiles may not map well to the **number of SMs**.
2. Tile shapes may fail to align with memory "burst sections", reducing the **effectiveness of coalesced accesses**.

#### 4. Operator fusion
Many operations require temporarily keeping data in on-chip memory (SRAM).
What happens when multiple operations are applied to the same small piece of data?
* **Naive:** load data from DRAM into SRAM, apply one operation, write the result
  back to DRAM; repeat this process for each subsequent operation.
* **Fused:** load data from DRAM into SRAM once, apply all operations sequentially
  while the data stays on-chip, then write the final result back to DRAM.

The difference lies in how many times data is transferred between DRAM and SRAM.
The illustration below, adapted from a detailed [blog post](https://horace.io/brrr_intro.html) by Horace He, shows this effect visually. I highly recommend reading this post for deeper dive.

<img src="/assets/img/triton/fused_kernel.png" alt="Img.4: Operator fusion" style="width:100%;">

With operator fusion, DRAM–SRAM transfers are reduced from $$O(n)$$ to $$O(1)$$, where $$n$$
is the number of operations.

#### 5. Recomputation
This optimization is closely related to operator fusion, but is more generally motivated by a key property of modern deep learning workloads: they are **often memory-bound rather than compute-bound**. In such cases, it can be beneficial to trade memory traffic for additional computation.

The idea of recomputation is to **avoid storing certain intermediate results** during the forward pass and instead recompute them when needed during the backward pass. This **reduces data movements** at the cost of extra compute, which is often a favorable tradeoff on modern GPUs. Below is a simple example for $$\text{out} = \sigma(\sigma(\sigma(x)))$$. 

<img src="/assets/img/triton/recomputation.png" alt="Img.5: Recomputation" style="width:100%;">

Instead of performing $$8$$ ($$4$$ writes + $$4$$ reads) memory operations, **recomputation reduces** this to $$5$$ ($$3$$ writes + $$2$$ reads) by recomputing intermediate activations on demand.

#### 6. Low precision computation
The core idea is simple: moving fewer bits is faster. Attention kernels are often memory-bound, so reducing tensor precision significantly improves speed and cache efficiency.

Importantly, low precision usually has negligible impact on model quality **if accumulation is handled carefully**. In practice, this means separating storage precision from accumulation precision.

**Operations that can use 16-bit (FP16/BF16) storage and compute:**
* matrix multiplications
* most pairwise operations (e.g. relu, tanh, multiplication)

**Operations that need more precision (FP32/FP16):**
* adding small values to large sums - can lead to rounding errors
* reduction operations (softmax, normalization)

**Operations that need more range (FP32/BF16):**
* pointwise operations where $$( \lVert f(x) \rVert \gg \lVert x \rVert )$$, i.e. the function significantly amplifies magnitude (e.g. $\text{exp, pow}$)

Below is an illustration of how a Tensor Core performs low-precision matrix multiplication. For compactness, the example shows the dot product between a row of matrix $$A$$ and a column of matrix $$B$$. The input operands are stored in FP16, while the accumulation is performed in FP32.

<img src="/assets/img/triton/tensor_cores.png" alt="Img.6: Tensor core" style="width:100%;">

FlashAttention-2 relies on this exact tradeoff: $$Q$$, $$K$$, and $$V$$ are stored and multiplied in low precision, while softmax statistics are accumulated in FP32.

#### 7. Control divergence {#control-divergence}
Threads within a single warp execute instructions in lockstep. When control flow diverges
(e.g. due to conditional branches), the warp must serialize the different execution paths, masking inactive threads. This **reduces parallelism** and slows down execution. 

Below is an illustration of how control divergence unfolds over time. Different colors represent different instruction paths executed by threads within the same warp.

<img src="/assets/img/triton/divergence_execution.png" alt="Img.7: Divergence of instructions within single warp" style="width:100%;">

---

## II. FlashAttention-2 in Triton: Kernel Design Walkthrough
#### 1. Why Triton?
Before **2021**, writing high-performance GPU kernels typically **required using CUDA**, a C++-based language with a steep learning curve. At the same time, most of the AI research community works primarily in Python, creating a **significant gap** between research code and production-grade GPU implementations.

To bridge this gap, [Philippe Tillet](https://scholar.google.com/citations?user=SQfo7UgAAAAJ) began developing Triton during his PhD at Harvard and later continued the work at OpenAI. Triton was officially [released](https://openai.com/index/triton/) as open source in 2021 as a Python-based language for writing custom GPU kernels.

In CUDA, developers must explicitly manage low-level details such as memory coalescing, shared memory usage, and scheduling within Streaming Multiprocessors (SMs). In Triton, this **low-level** machinery does **not need to be written manually**, but the programmer still explicitly controls data layout, tiling, and parallelism directly in the kernel. This makes Triton particularly convenient for research engineers to prototype and iterate on custom GPU kernels, which would be prohibitively slow to develop and maintain directly in CUDA.

#### 2. Execution Model in Triton
A useful mental model for programming in Triton is to separate the implementation into **two parts**:
- a **Triton kernel**, written as a Python function, which executes a single program instance (typically mapped to a block of work on an SM) and operates on pointers and strides;
- a **Python launch wrapper**, which prepares inputs, configures the execution grid, and calls the Triton kernel.

In this implementation, I wrap both the forward and backward kernels inside a custom PyTorch `torch.autograd.Function`. This requires defining explicit `forward` and `backward` methods, while giving full control over memory allocation, kernel launches, and recomputation strategies.

A high-level sketch of the structure is shown below.


```python
class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, q_tile=128, k_tile=128, num_warps=4, num_stages=2):
        # Get dimension
        B, H, N_QUERIES, D_MODEL = Q.shape
        _, H_kv, N_KEYS, _ = K.shape

        ...

        # Initialize empty result tensor, logsumexps 
        O = torch.zeros((B, H, N_QUERIES, D_MODEL), device = Q.device, dtype=Q.dtype)
        L = torch.zeros((B, H, N_QUERIES), device=Q.device, dtype=torch.float32)

        # Strides
        stride_qb, stride_qh, stride_qs, stride_qd = Q.stride()
        ...
        
        # Run kernel
        scale = 1 / (D_MODEL ** 0.5)
        grid = (B, H, triton.cdiv(N_QUERIES, q_tile))
        flashattn_fwd[grid](
            Q_ptr = Q, K_ptr = K, V_ptr = V, O_ptr = O, L_ptr = L,
            stride_qb = stride_qb, stride_qh = stride_qh, stride_qs = stride_qs, stride_qd = stride_qd,
            ...
        )

        # Cache vars necessary for backward
        ctx.save_for_backward(L, Q, K, V, O)
        ...

        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        D = (O * dO).sum(axis = -1)
        ...

        # Run kernel
        grid = (B, H, triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE))
        flashattn_bcwd[grid](...)

        # NOTE: We must return derivatives of the same variables that we use as input in forward method
        return dQ, dK, dV, None, None, None, None, None
```

Here, the **Triton kernels** implement the core **attention logic**, while the **Python code** is responsible for defining the **execution grid**, passing tensor strides, and caching only the tensors required by the backward pass.

#### 3. Forward Pass: Tiled Attention with Online Softmax
We first recall the **naive attention** computation.

Given queries $$\mathbf{Q} \in \mathbb{R}^{N_q \times d}$$, keys $$\mathbf{K} \in \mathbb{R}^{N_k \times d}$$, and values $$\mathbf{V} \in \mathbb{R}^{N_k \times d}$$:
1. Compute attention scores: $$\mathbf{S = QK^T / \sqrt{d}}$$.
2. Apply row-wise softmax: $$\mathbf{P_{i, } = softmax(S_{i,})}$$.
3. Compute output: $$\mathbf{O = PV}$$.

This formulation requires **materializing** the matrices $\mathbf{S}$ and $\mathbf{P}$, both of size $\mathbb{R}^{N_q \times N_k}$, which is prohibitively **expensive** for long sequences and **inefficient** due to excessive memory traffic.

To implement the forward pass of FlashAttention, we rely on [tiling](#tiling), discussed earlier, together with **online softmax**, which allows softmax to be computed incrementally. A detailed derivation of online softmax is provided in the [supplementary post](http://timashov.ai/blog/2025/attention-backward/).

Let $$\mathbf{Q_q} \in \mathbb{R}^{T_q \times d}, \mathbf{K_k}, \mathbf{V_k} \in \mathbb{R}^{T_k \times d}$$ denote tiles of $$\mathbf{Q}, \mathbf{K}, \mathbf{V}$$.

Maintain per-row running state:
$$\mathbf{m} \in \mathbb{R}^{T_q}, \ \mathbf{L_q} \in \mathbb{R}^{T_q}, \ \mathbf{O_q} \in \mathbb{R}^{T_q \times d}$$.

Initialize:
$$\mathbf{m} = -\infty, \ \mathbf{L_q} = 0, \ \mathbf{\hat{O}} = 0$$.

For each tile ($$\mathbf{K_k}, \mathbf{V_k}$$):

1. Compute tiled scores: $$\mathbf{S_{q,k}} = \mathbf{Q_q K^T_k} / \mathbf{\sqrt{d}}$$.
2. Update the running maximum: $$\mathbf{m}_\text{new} = \max(\mathbf{m}, \text{rowmax}(\mathbf{S_{q,k}}))$$.
3. Compute exponentiated scores: $$\mathbf{P_{q,k}} = \exp(\mathbf{S_{q,k}} - \mathbf{m}_\text{new})$$.
4. Update the normalization term: $$\mathbf{L_q} = \mathbf{L_q} \cdot \exp(\mathbf{m} - \mathbf{m}_\text{new}) + \text{rowsum}(\mathbf{P_{q,k}})$$.
5. Update the output: $$\mathbf{O_q} = \mathbf{O_q} \cdot \exp(\mathbf{m} - \mathbf{m}_\text{new}) + \mathbf{P_{q,k}V_k}$$.
6. Update the running maximum: $$\mathbf{m} \leftarrow \mathbf{m}_\text{new}$$.

After all tiles are processed, the final output is: $$\mathbf{O_q} \leftarrow \mathbf{O_q} / \mathbf{L_q}$$.

Crucially, this algorithm never materializes $\mathbf{S}$ or $\mathbf{P}$. The largest intermediate matrix has size $T_q \times T_k$ - the size of a single tile.

In code, one iteration over a key/value tile looks as follows:
```python
# Load the current tiles of K, V
K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

# Compute scaled tile score
S = tl.dot(Q, tl.trans(K)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

# Compute temporary running features
m_hat = S.max(axis = -1) # (Q_TILE_SIZE, )
m_new = tl.maximum(m, m_hat) # (Q_TILE_SIZE,)

# Compute exponent of score and substract max values 
P_j = tl.exp(S - m_new[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)

# Update L
mult = tl.exp(m - m_new) # (Q_TILE_SIZE,)
L = L * mult + tl.sum(P_j, axis = -1) # (Q_TILE_SIZE,)

# Compute running output
O = O * mult[:, None] + tl.dot(P_j.to(V.dtype), V) # (Q_TILE_SIZE, D)

# Update running max
m = m_new # (Q_TILE_SIZE,)
```

#### 4. Backward Pass: Recomputation Instead of Materialization
I begin by recalling the standard backward equations for attention:

$$
\mathbf{dV = \textcolor{red}{P^T} dO}
$$

$$
\mathbf{dP = dOV^T}
$$

$$
\mathbf{dS_{i,} = dsoftmax(dP_{i,}) = dP_{i,}(diag(P_{i,})-P_{i,}^T P_{i,})}
$$

$$
\mathbf{dQ = dSK / \sqrt{d}}
$$

$$
\mathbf{dK = dS^T Q / \sqrt{d}}
$$

Derivations of these expressions, including the softmax gradient, are provided in the [supplementary post](http://timashov.ai/blog/2025/attention-backward/).

As in the forward pass, the naive backward computation depends on materializing large intermediate tensors. In particular, both $\mathbf{dV}$ and $\mathbf{dS}$ depend on the attention matrix $\mathbf{P} \in \mathbb{R}^{N_q \times N_k}$, which is infeasible to store and move between global and shared memory.

The important observation is that the softmax gradient depends on the row-wise quantity $$\mathrm{rowsum}(\mathbf{P} \odot \mathbf{dP})$$, and that this term can be equivalently computed as $$\mathrm{rowsum}(\mathbf{P} \odot \mathbf{dP}) = \mathrm{rowsum}(\mathbf{O} \odot \mathbf{dO})$$, which involves only tensors available from the forward pass and upstream gradients.

This identity enables a **recomputation-based backward pass**, avoiding materialization of $\mathbf{P}$. Specifically, I reuse:
- the log-sum-exp values $\mathbf{L}$ from the forward pass,
- the output $\mathbf{O}$,
- the upstream gradient $\mathbf{dO}$.

Define:

$$\mathbf{D} = \mathrm{rowsum}(\mathbf{O} \odot \mathbf{dO})$$.

For each key/value tile ($$\mathbf{K_k}$$, $$\mathbf{V_k}$$):
1. Recompute tiled scores: $$\mathbf{S_{q,k}} = \mathbf{Q_q K^T_k} / \mathbf{\sqrt{d}}$$.
2. Recompute tiled attention weights: $$\mathbf{P_{q,k}} = \exp(\mathbf{S_{q,k}} - \mathbf{L_q})$$.
3. Compute $\mathbf{dP_{q,k}} = \mathbf{dO_q} \mathbf{V^T_k}$.
4. Compute: $$\mathbf{dS_{q,k}} = \mathbf{P_{q,k}} \odot (\mathbf{dP_{q,k}} - \mathbf{D_q})$$.
5. Accumulate $\mathbf{dQ_q}$ across $$\mathbf{k}$$ tiles.
6. Compute $\mathbf{dK_k}$ and $\mathbf{dV_k}$ as in the standard backward equations.

This backward pass mirrors the forward computation: large intermediate tensors are never stored, and all operations are performed in tiles.

In code, one iteration over a key/value tile looks as follows:
```python
# Load the current tiles of K, V
K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D_MODEL)
V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D_MODEL)

# 1. Recompute scaled tile score
S = tl.dot(Q, tl.trans(K)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

...

# 2. Recompute exponents of scores
P = tl.exp(S.to(tl.float32) - L[:, None])

# 3. Compute dP
dP = tl.dot(dO.to(tl.float32), tl.trans(V).to(tl.float32))

# 4. Compute dS
dS = P * (dP - D[:, None]) * scale

# 5a. Compute dV
dV = tl.dot(tl.trans(P), dO.to(tl.float32))

# 5b. Compute dQ, dK
dQ += tl.dot(dS, K.to(tl.float32))
dK = tl.dot(tl.trans(dS), Q.to(tl.float32))

...
```

---

## III. Performance Analysis

---

## IV. Key Takeaways for Writing Fast Attention Kernels


