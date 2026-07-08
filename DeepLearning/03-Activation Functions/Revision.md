# Activation Functions — Revision Notes

> **Module:** 03 - Activation Functions
>
> **Purpose:** 10-minute revision before interviews.

---

# Learning Journey

Understanding activation functions is a journey of solving one limitation after another.

```text
Perceptron
      │
      ▼
Step Function
      │
      ▼
Sigmoid
      │
      ▼
ReLU
      │
      ▼
Leaky ReLU
      │
      ▼
GELU
      │
      ▼
Softmax
      │
      ▼
Cross Entropy
```

Every new activation function was introduced to solve a problem left by the previous one.

---

# Activation Function Comparison

| Activation | Output Range | Differentiable | Main Use | Major Limitation |
|------------|-------------|---------------|-----------|------------------|
| Step Function | {0,1} | ❌ No | Original Perceptron | Cannot use Gradient Descent |
| Sigmoid | (0,1) | ✅ Yes | Binary Classification | Vanishing Gradient |
| ReLU | [0,∞) | Almost | Hidden Layers | Dead ReLU Problem |
| Leaky ReLU | (-∞,∞) | Yes | Hidden Layers | Small negative output |
| GELU | Smooth | Yes | Transformers / GPT | Computationally expensive |
| Softmax | (0,1) | Yes | Multi-Class Output Layer | Only for mutually exclusive classes |

---

# Why Did We Keep Inventing New Activation Functions?

## Step Function

Problem Solved

- Binary decision (Fire / Don't Fire)

Problem

- Not differentiable
- Impossible to train using Gradient Descent

↓

Need a smooth activation.

---

## Sigmoid

Formula

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Advantages

- Smooth
- Differentiable
- Outputs probabilities

Problems

- Vanishing Gradient
- Not zero-centered
- Slow convergence

↓

Need a better hidden-layer activation.

---

## ReLU

Formula

$$
ReLU(x)=\max(0,x)
$$

Advantages

- Fast
- Simple
- Reduces Vanishing Gradient
- Sparse activations

Problem

- Dead ReLU

↓

Need neurons to recover even for negative inputs.

---

## Leaky ReLU

Formula

$$
LeakyReLU(x)=
\begin{cases}
x,&x>0\\
\alpha x,&x\le0
\end{cases}
$$

Advantages

- Solves Dead ReLU
- Allows gradient flow for negative inputs

Problem

- Choice of α is heuristic
- Still not perfectly smooth

↓

Need a smoother activation.

---

## GELU

Approximate Formula

$$
GELU(x)
=
0.5x
\left(
1+
\tanh
\left(
\sqrt{\frac{2}{\pi}}
\left(
x+0.044715x^3
\right)
\right)
\right)
$$

Idea

Instead of switching neurons completely ON or OFF,

GELU gradually decides

> "How much should this neuron contribute?"

Used in

- BERT
- GPT
- Gemini
- Claude
- Most modern Transformers

---

## Softmax

Formula

$$
Softmax(x_i)=
\frac{e^{x_i}}
{\sum_j e^{x_j}}
$$

Purpose

Converts logits into a probability distribution.

Properties

- Every probability > 0
- Sum = 1
- Classes compete with one another

Used only in

- Multi-class classification
- Output layer

---

## Cross Entropy

Purpose

Measures how close the predicted probability distribution is to the true distribution.

Simplified Formula

$$
Loss=-\log(P(\text{Correct Class}))
$$

High confidence + Correct Prediction

↓

Small Loss

Low confidence or Wrong Prediction

↓

Large Loss

---

# Softmax vs Sigmoid

| Sigmoid | Softmax |
|----------|----------|
| Binary Classification | Multi-Class Classification |
| Independent outputs | Competing outputs |
| One neuron | Multiple output neurons |
| Output between 0 and 1 | Probabilities sum to 1 |

---

# ReLU vs GELU

| ReLU | GELU |
|------|------|
| Hard threshold | Smooth transition |
| ON / OFF behaviour | Gradual gating |
| Very fast | Slightly slower |
| CNNs and classic DNNs | Transformers |

---

# Softmax Pipeline

```text
Input

↓

Hidden Layers

↓

Final Linear Layer

↓

Logits

↓

Softmax

↓

Probabilities

↓

Cross Entropy

↓

Loss

↓

Gradient Descent

↓

Weight Update
```

---

# Interview Rapid Fire

### Why can't Step Function be used in modern neural networks?

It is not differentiable, making Gradient Descent impossible.

---

### Why did Sigmoid replace Step Function?

Because it is smooth and differentiable.

---

### Why did ReLU replace Sigmoid?

To reduce the Vanishing Gradient problem and speed up training.

---

### Why was Leaky ReLU introduced?

To solve the Dead ReLU problem.

---

### Why do Transformers use GELU?

Because it provides smooth probabilistic gating, leading to better optimization.

---

### What are logits?

Raw outputs of the final linear layer before Softmax.

---

### Why do we need Softmax?

To convert logits into probabilities.

---

### Why can't ReLU replace Softmax?

ReLU processes neurons independently and cannot produce a probability distribution.

---

### What does Cross Entropy measure?

The difference between the predicted probability distribution and the true distribution.

---

### Why are Softmax and Cross Entropy used together?

Softmax produces probabilities.

Cross Entropy evaluates those probabilities.

---

# Common Interview Mistakes

❌ Softmax is used in hidden layers.

✅ Softmax is almost always used only in the output layer.

---

❌ ReLU outputs probabilities.

✅ ReLU only introduces non-linearity.

---

❌ Logits are probabilities.

✅ Logits are raw scores before Softmax.

---

❌ Cross Entropy converts logits into probabilities.

✅ Softmax converts logits into probabilities.

Cross Entropy measures the quality of those probabilities.

---

# 60-Second Mind Map

```text
Artificial Neuron
        │
        ▼
Perceptron
        │
        ▼
Activation Function
        │
        ├── Step Function
        ├── Sigmoid
        ├── ReLU
        ├── Leaky ReLU
        ├── GELU
        └── Softmax
                │
                ▼
Probability Distribution
                │
                ▼
Cross Entropy
                │
                ▼
Loss
                │
                ▼
Gradient Descent
```

---

# Key Takeaways

✅ Every new activation function solved a limitation of the previous one.

✅ ReLU became the standard activation for deep neural networks.

✅ GELU is the preferred activation in modern Transformer architectures.

✅ Softmax is a specialized output-layer activation for multi-class classification.

✅ Cross Entropy evaluates how good the predicted probabilities are.

✅ Activation functions make deep learning possible by introducing non-linearity.

---

# One-Line Revision

> **Step → Sigmoid → ReLU → Leaky ReLU → GELU → Softmax → Cross Entropy**

Remember the evolution, not just the formulas.