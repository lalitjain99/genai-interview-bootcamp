# Mini-Batch Gradient Descent

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- Why is Mini-Batch Gradient Descent needed?
- How does it combine the advantages of Batch GD and SGD?
- What is a mini-batch?
- How are gradients computed in Mini-Batch GD?
- Why is Mini-Batch GD the industry standard?
- How does batch size affect training?
- Why do GPUs prefer Mini-Batch training?

---

# Chapter 1 — The Best of Both Worlds

We have now studied two optimization methods.

## Batch Gradient Descent

```text
Entire Dataset

↓

Compute Average Gradient

↓

One Weight Update
```

Advantages:

- Accurate gradients
- Stable optimization

Disadvantages:

- Very slow
- High memory usage

---

## Stochastic Gradient Descent

```text
One Sample

↓

Compute Gradient

↓

Update Weights
```

Advantages:

- Very fast updates
- Low memory usage

Disadvantages:

- Very noisy
- Unstable optimization

---

Neither approach is ideal.

So engineers asked:

> **Can we take a small group of samples instead of one sample or the entire dataset?**

The answer is

> **Mini-Batch Gradient Descent**

---

# Chapter 2 — What is a Mini-Batch?

Instead of processing

```text
Entire Dataset
```

or

```text
One Sample
```

we process a **small subset**.

For example,

```text
Dataset

↓

Batch Size = 32
```

The dataset is divided into small groups.

```text
Sample 1

Sample 2

...

Sample 32
```

This group of 32 samples is called a **Mini-Batch**.

Common batch sizes are:

```text
16

32

64

128

256
```

---

# Chapter 3 — How Mini-Batch Gradient Descent Works

Suppose we have

```text
10,000 Training Samples
```

and

```text
Batch Size = 100
```

The dataset is divided into

```text
100 Mini-Batches
```

Training proceeds as follows:

```text
Mini-Batch 1

↓

Forward Pass

↓

Compute Loss

↓

Average Gradient

↓

Update Weights
```

Then

```text
Mini-Batch 2

↓

Forward Pass

↓

Compute Loss

↓

Average Gradient

↓

Update Weights
```

This continues until all mini-batches have been processed.

One complete pass through all mini-batches is called **one epoch**.

---

# Chapter 4 — Why Mini-Batch is Better

Mini-Batch Gradient Descent combines the strengths of Batch GD and SGD.

It uses more than one sample,

so the gradient is much more reliable than SGD.

But it does not wait for the entire dataset,

so updates happen much more frequently than Batch GD.

```text
Batch GD

↓

Very Accurate

Very Slow
```

```text
SGD

↓

Very Fast

Very Noisy
```

```text
Mini-Batch GD

↓

Fast

Stable

Efficient
```

It provides the best practical balance between speed and stability.

---

# Chapter 5 — A Classroom Analogy

Imagine a teacher with 1,000 students.

### Batch Gradient Descent

The teacher waits until all 1,000 students complete the exam.

Then changes the teaching strategy.

---

### Stochastic Gradient Descent

The teacher changes the teaching strategy after every single student.

This reacts quickly,

but may overreact to one unusual student.

---

### Mini-Batch Gradient Descent

The teacher divides the class into groups of 25 students.

After each group,

the teacher evaluates the average performance and makes a small adjustment.

```text
25 Students

↓

Average Performance

↓

Adjust Teaching

↓

Next 25 Students
```

This is both stable and responsive.

---

# Chapter 6 — Batch Size Matters

The batch size directly affects training behavior.

### Very Small Batch

```text
Batch Size = 2
```

Characteristics:

- Frequent updates
- More noise
- Less memory
- Faster individual updates

---

### Very Large Batch

```text
Batch Size = 2048
```

Characteristics:

- Smoother gradients
- Less noise
- Higher memory usage
- Fewer updates

---

There is no universally best batch size.

The optimal choice depends on:

- Dataset
- Model size
- GPU memory
- Training objectives

---

# Chapter 7 — Why GPUs Love Mini-Batches

Modern GPUs are designed for **parallel computation**.

Suppose we process

```text
One Sample
```

Many GPU cores remain idle.

Now suppose we process

```text
Batch Size = 128
```

The GPU can perform many matrix multiplications simultaneously.

```text
128 Samples

↓

Parallel Matrix Operations

↓

GPU Utilization Increases

↓

Training Becomes Faster
```

Mini-Batch Gradient Descent efficiently utilizes GPU hardware, making it the preferred approach for deep learning.

---

# Chapter 8 — Why Every Modern LLM Uses Mini-Batches

Imagine training GPT on

```text
10 Trillion Tokens
```

Using Batch GD:

```text
Process Everything

↓

One Update
```

Impossible.

Using SGD:

```text
One Token

↓

One Update
```

Too noisy.

Instead,

tokens are grouped into mini-batches.

```text
Mini-Batch of Tokens

↓

Forward Pass

↓

Loss

↓

Average Gradient

↓

Weight Update
```

This process repeats across millions of mini-batches during training.

---

# Chapter 9 — Comparison

| Property | Batch GD | SGD | Mini-Batch GD |
|-----------|----------|-----|---------------|
| Samples per Update | Entire Dataset | 1 | Small Batch |
| Speed | Slow | Fast | Fast |
| Gradient Stability | Excellent | Poor | Good |
| Memory Usage | High | Low | Moderate |
| GPU Utilization | Poor | Poor | Excellent |
| Used in Modern Deep Learning | No | Rarely | Yes |

---

# Chapter 10 — Looking Ahead

Mini-Batch Gradient Descent solved many practical problems.

However,

another question remained.

Even with Mini-Batches,

training could still oscillate,

especially in steep regions of the loss surface.

Engineers asked:

> **Can the optimizer remember previous gradients and build momentum?**

That question leads to our next optimizer:

```text
Momentum
```
---

# Engineer Q&A

## Q1. Why was Mini-Batch Gradient Descent invented?

We already had two optimization algorithms:

- Batch Gradient Descent
- Stochastic Gradient Descent

Why wasn't either one sufficient?

### Answer

Both Batch Gradient Descent and Stochastic Gradient Descent have significant drawbacks.

**Batch Gradient Descent**

Advantages:

- Accurate gradients
- Stable optimization

Disadvantages:

- Very slow
- High memory usage

---

**Stochastic Gradient Descent**

Advantages:

- Fast weight updates
- Low memory usage

Disadvantages:

- Very noisy gradients
- Unstable optimization

Mini-Batch Gradient Descent was introduced to combine the strengths of both approaches.

Instead of processing the **entire dataset** or **one sample at a time**, it processes a **small group of samples**.

```text
Small Group of Samples

↓

Average Gradient

↓

Weight Update
```

This produces smoother gradients than SGD while updating weights much more frequently than Batch Gradient Descent.

---

## Q2. Suppose a dataset contains 10,000 samples and the batch size is 100.

Answer the following:

- How many mini-batches are created?
- How many weight updates occur in one epoch?
- How many forward passes occur?
- How many backward passes occur?

### Answer

Number of mini-batches:

```text
10,000 / 100 = 100 Mini-Batches
```

Each mini-batch performs:

```text
Forward Pass

↓

Loss Computation

↓

Backward Pass

↓

Weight Update
```

Therefore, during one epoch:

- Mini-Batches = **100**
- Forward Passes = **100**
- Backward Passes = **100**
- Weight Updates = **100**

Each mini-batch contributes one complete training step.

---

## Q3. Compare different batch sizes.

### Option A

```text
Batch Size = 1
```

### Option B

```text
Batch Size = 64
```

### Option C

```text
Batch Size = Entire Dataset
```

Discuss the trade-offs.

### Answer

### Batch Size = 1 (SGD)

Advantages:

- Very frequent updates
- Low memory usage

Disadvantages:

- Very noisy gradients
- Poor optimization stability
- Poor GPU utilization

---

### Batch Size = 64 (Mini-Batch GD)

Advantages:

- Smoother gradients
- Less noisy optimization
- Moderate memory usage
- Excellent GPU utilization
- Good balance between speed and stability

This is why Mini-Batch Gradient Descent is the preferred choice for modern deep learning.

---

### Batch Size = Entire Dataset (Batch GD)

Advantages:

- Very accurate gradients
- Stable optimization

Disadvantages:

- Very slow updates
- High memory usage
- Impractical for modern large-scale datasets because the entire dataset usually cannot fit into GPU memory

---

## Q4. Why does Mini-Batch Gradient Descent produce a smoother optimization path than SGD?

### Answer

Every training sample produces its own gradient.

Some gradients point slightly left.

Others point slightly right.

Some noisy samples may produce gradients that differ significantly.

SGD immediately follows each individual gradient.

```text
One Sample

↓

Gradient

↓

Immediate Update
```

Mini-Batch Gradient Descent first averages the gradients from multiple samples.

```text
64 Samples

↓

Average Gradient

↓

Weight Update
```

Suppose:

```text
60 Samples

↓

Gradient → Left
```

and

```text
4 Noisy Samples

↓

Gradient → Right
```

The average gradient still points mostly to the left.

The noisy gradients do not disappear, but their influence becomes much smaller.

This produces a much smoother optimization path than pure SGD.

---

## Q5. What happens if the batch size keeps increasing?

Suppose the batch size changes as follows:

```text
32

↓

64

↓

128

↓

256

↓

512

↓

1024

↓

Entire Dataset
```

Which optimization algorithm does it eventually become?

### Answer

As the batch size increases, Mini-Batch Gradient Descent gradually behaves more like Batch Gradient Descent.

When the batch size equals the entire dataset:

```text
Entire Dataset

↓

Average Gradient

↓

One Weight Update
```

Only one weight update occurs per epoch.

This is exactly the definition of Batch Gradient Descent.

Therefore,

```text
Mini-Batch Gradient Descent

↓

Batch Size = Entire Dataset

↓

Batch Gradient Descent
```

---

# Engineering Discussion

## Why isn't the largest possible batch size always the best?

A common misconception is:

> "Larger batches always produce better training."

While larger batches generate smoother gradients, they also reduce the number of weight updates performed during each epoch.

For example:

```text
Dataset = 10,000 Samples
```

If

```text
Batch Size = 100
```

then

```text
100 Weight Updates per Epoch
```

However,

if

```text
Batch Size = 10,000
```

then

```text
1 Weight Update per Epoch
```

Although the gradient is very accurate, the model receives far fewer opportunities to learn during an epoch.

Mini-Batch Gradient Descent strikes a practical balance by providing both reasonably accurate gradients and frequent weight updates.

---

# Future Engineering Note

During this lecture, an important engineering question naturally arises:

> **Why can't we simply keep increasing the batch size if we have enough GPU memory?**

Answering this requires understanding topics such as:

- GPU architecture
- VRAM
- Parallel computation
- Matrix multiplication
- Memory bandwidth
- CUDA
- Training vs Inference

These concepts will be covered later in a dedicated module:

```text
AI Hardware Fundamentals
```

Understanding hardware is essential for designing scalable AI systems and making informed engineering decisions at the Staff Engineer or AI Architect level.
---

# Key Takeaways

- Mini-Batch Gradient Descent processes small groups of training samples.
- It combines the stability of Batch GD with the speed of SGD.
- Weight updates occur after each mini-batch.
- Batch size is an important hyperparameter.
- Mini-Batches efficiently utilize GPU parallelism.
- Nearly all modern deep learning models are trained using Mini-Batch Gradient Descent.