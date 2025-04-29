---
layout: post
title: "How Computers Store Data in Memory: Brief Intro"
date: 2025-04-26 09:00:00
description: fp32, fp16, and more
tags: pre-dl
categories: cs336
# thumbnail: assets/img/9.jpg
images:
  compare: true
  slider: true
---


## Introduction
When **we communicate** with each other, we use complex **natural languages** like English, Portuguese, or Russian. The language we use depends on our location or the community around us. When we see something, we perceive information visually through our eyes and brain. And when we count numbers, most people are familiar with one system — the **decimal system**, where each position can be represented by one of 10 digits: `0, 1, 2, ..., 9`. So, at least three independent ways exist to perceive and exchange information.  
**Computers**, however, operate very differently — and in some sense, much more simply. They do not have a native natural language or vision system at their core. Instead, they use only the **binary system**, composed of `0` and `1`. Each "box" of information can contain either a 0 or a 1, and this unit is called a **bit**. Historically, 8 bits make up a **byte**.


<img src="/assets/img/bits_bytes.png" alt="Img.1: Explanation of Bits and Bytes" style="width:100%;">

Higher-level constructs, such as **Natural Language Models** (*LLaMA, ChatGPT, Grok, etc.*) and **Vision Systems** (*YOLO, Faster R-CNN, Stable Diffusion, etc.*), are built on top of this fundamental binary representation.


Understanding how computers "speak" at the basic level is critical for building intuition in Deep Learning. In this post, I learn how computers represent and store information, how it connects to data types, and why these foundations matter when designing and optimizing AI models.


---

## Decimal vs Binary
The **decimal system** is the numeral system we use in daily life. Each digit can take one of *10* different values: `0, 1, 2, ..., 9`, ordered naturally from smallest to largest. To construct numbers, we use powers of *10*: when increasing past *9*, we reset the digit to *0* and add *1* to the next higher place value. The **binary system** works similarly — but instead of *10* possible values, each digit (bit) can only be `0` or `1`. Here, the base is *2*, and each digit represents a power of *2*.

<img src="/assets/img/nums_seq.png" alt="Img.2: Sequence of numbers in Decimal and Binary systems" style="width:100%;">


Fractional numbers follow the same idea: each digit after the floating point represents a negative power of the base - *10* for decimal numbers, and *2* for binary numbers. The algorithm for **converting a fractional number from decimal to binary** is iterative and slightly different for the integer and fractional parts:
* For the **integer part**, divide the number by *2*, record the remainder (*0* or *1*), and continue dividing the quotient by *2* **until** it becomes *0*.
* For the **fractional part**, multiply the fraction by *2*, record the integer part (*0* or *1*), and repeat the process with the remaining fractional part. The process **stops when** the fraction becomes 0 or when the desired precision is reached.

Below is the step-by-step calculation for the number `13.875`:

<img src="/assets/img/binary_repr.png" alt="Img.3: Representation of fractional numbers in Decimal and Binary systems" style="width:100%;">

We're used to the decimal system; however, the binary system follows the exact same logic — just with *2* symbols instead of *10*. Inside a computer, tiny transistors act as switches that can be either ON or OFF, determined by voltage levels. It’s much easier and more reliable to distinguish just two states — high vs. low voltage — than to detect multiple. This simplicity and robustness is **why all information in computers is stored in binary**.


## Main Data Types

At the core of computing, we work with a few basic types:

- **Integer (`int`)**: Whole numbers, like `-1`, `0`, `42`.
- **Floating-point (`float`)**: Numbers with decimals, like `3.14`, `-0.001`.
- **Boolean (`bool`)**: `True` or `False`, used in logical decisions.
- **Character (`char`)**: Single characters like `'a'`, `'Z'`, `'#'`.
- **String (`str`)**: Sequences of characters like `"Hello"`.

Each type has a size and memory footprint. For example:

- `int32` (32-bit integer) or `float64` (64-bit floating-point).

---

## Data Types Used in Deep Learning

Deep learning relies heavily on numerical tensors — multi-dimensional arrays where **type precision** matters for memory and speed.

Common types:

- **`float32`**: 32-bit floating point — standard for most models.
- **`float16`**: Half-precision — used for faster computation (e.g., NVIDIA Tensor Cores).
- **`bfloat16`**: Brain Floating Point, optimized for AI workloads.
- **`int8`**: 8-bit integers — used in quantized models to reduce model size.

Choosing the right type can influence model **speed**, **accuracy**, and **training stability**.

---

## Quick Back-of-Envelope Calculation

Suppose you have a tensor with shape `(batch_size=64, channels=3, height=224, width=224)`, typical for an image classification model.

How much memory would it consume with different types?

- Number of elements:  
  `64 × 3 × 224 × 224 = 9,633,792`

- Memory usage:
  - `float32` (4 bytes per value):  
    `9,633,792 × 4 = ~38.5 MB`
  - `float16` (2 bytes per value):  
    `9,633,792 × 2 = ~19.2 MB`
  - `int8` (1 byte per value):  
    `9,633,792 × 1 = ~9.6 MB`

**Notice** how changing the data type immediately halves or quarters the memory footprint.

---

## Conclusion

Understanding data types is the foundation of building efficient AI systems.  
It affects not only how we design models but also how fast and how large they can be.

In future posts, we'll dive deeper into **precision trade-offs** and **mixed-precision training** strategies.




## Image Slider

This is a simple image slider. It uses the [Swiper](https://swiperjs.com/) library. Check the [examples page](https://swiperjs.com/demos) for more information of what you can achieve with it.

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/9.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/7.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/8.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/10.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/12.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>

## Image Comparison Slider

This is a simple image comparison slider. It uses the [img-comparison-slider](https://img-comparison-slider.sneas.io/) library. Check the [examples page](https://img-comparison-slider.sneas.io/examples.html) for more information of what you can achieve with it.

<img-comparison-slider>
  {% include figure.liquid path="assets/img/prof_pic.jpg" class="img-fluid rounded z-depth-1" slot="first" %}
  {% include figure.liquid path="assets/img/prof_pic_color.png" class="img-fluid rounded z-depth-1" slot="second" %}
</img-comparison-slider>
