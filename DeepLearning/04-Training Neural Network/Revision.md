# Revision — Training Neural Networks

---

# The Big Picture

A neural network is not born intelligent.

Initially, all of its weights are random.

Because of these random weights, its predictions are also almost random.

The purpose of training is simple:

> **Gradually adjust the weights so that the network makes better predictions.**

Everything we learned in this module is simply one part of this learning process.

---

# Step 1 — The Forward Pass

Training always begins with an input.

Suppose we want to classify an image.

```text
Image

↓

Neural Network

↓

Prediction
```

Inside the neural network, each layer performs two operations.

```text
Linear Layer
(z = Wx + b)

↓

Activation Function
(ReLU / GELU / Sigmoid)

↓

Next Layer
```

The final layer usually applies **Softmax**, producing a probability distribution.

For example,

```text
Dog      = 2%

Cat      = 97%

Horse    = 1%
```

These probabilities represent the model's prediction.

This entire process is called the **Forward Pass**.

---

# Step 2 — Measure the Error

A prediction alone is not enough.

The network must know **how good or bad** its prediction is.

This is the job of the **Loss Function**.

For regression problems:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

For classification problems:

- Cross Entropy
- Binary Cross Entropy

Suppose the correct label is

```text
Cat
```

If the model predicts

```text
Cat = 97%
```

the loss is very small.

If the model predicts

```text
Cat = 2%
```

the loss is very large.

A good prediction produces a small loss.

A poor prediction produces a large loss.

---

# Step 3 — Why Cross Entropy Uses Log

Cross Entropy is designed to reward confident correct predictions and heavily punish confident mistakes.

Instead of using the probability directly,

it computes

$$
-\log(p)
$$

This has two important properties:

- If the correct class has a high probability, the loss is close to zero.
- If the correct class has a very low probability, the loss becomes very large.

This is exactly the behavior we want during training.

---

# Step 4 — The Computational Graph

While making the prediction,

the framework quietly records every operation.

```text
Input
   │
   ▼
Linear Layer
   │
   ▼
ReLU
   │
   ▼
Linear Layer
   │
   ▼
Softmax
   │
   ▼
Loss
```

This structure is called the **Computational Graph**.

It stores how every value depends on the previous one.

---

# Step 5 — Backpropagation

Now the model needs to answer an important question:

> **Which weights caused this loss?**

To answer this,

Backpropagation starts from the loss and moves backward through the Computational Graph.

```text
Loss

↑

Softmax

↑

Linear Layer

↑

ReLU

↑

Linear Layer
```

The goal is to compute how much each weight contributed to the final error.

---

# Step 6 — Chain Rule

The loss does not depend directly on the weights.

Instead,

each weight influences the next computation,

which influences the next,

until finally affecting the loss.

```text
Weight

↓

Weighted Sum

↓

Activation

↓

Prediction

↓

Loss
```

To measure the total effect,

Backpropagation applies the **Chain Rule**.

The Chain Rule tells us that the total effect of one variable on another is the **product of all the local effects along the path**.

This allows gradients to flow backward through the network.

---

# Step 7 — Derivatives and Gradients

A derivative measures how quickly something changes.

For a neural network,

we ask:

> **If a weight changes slightly, how much will the loss change?**

When there are many weights,

we compute a **partial derivative** for each one while keeping the others fixed.

The collection of all these partial derivatives forms the **Gradient Vector**.

```text
Gradient

=

[
∂L/∂w₁,

∂L/∂w₂,

∂L/∂w₃,

...
]
```

The gradient points in the direction of the **steepest increase** in the loss.

---

# Step 8 — Gradient Descent

Since the gradient points uphill,

we move in the opposite direction.

The update rule is

$$
w = w - \eta \nabla L
$$

where

- $w$ = weights
- $\eta$ = learning rate
- $\nabla L$ = gradient

Each update slightly changes the weights,

making future predictions a little better.

---

# Step 9 — Repeat

Training does not stop after one example.

The same process repeats again and again.

```text
Forward Pass

↓

Prediction

↓

Loss

↓

Backpropagation

↓

Gradient Computation

↓

Weight Update

↓

Repeat
```

Over thousands or even millions of iterations,

the loss gradually decreases,

and the model becomes more accurate.

---

# Step 10 — The Complete Learning Pipeline

Everything we've learned can be summarized in one diagram.

```text
Training Data
       │
       ▼
Forward Pass
       │
       ▼
Prediction
       │
       ▼
Loss Function
       │
       ▼
Computational Graph
       │
       ▼
Backpropagation
       │
       ▼
Chain Rule
       │
       ▼
Gradient Vector
       │
       ▼
Gradient Descent
       │
       ▼
Updated Weights
       │
       ▼
Repeat Until Convergence
```

Every neural network—from a simple perceptron to GPT—learns using this same fundamental pipeline.

The architecture may become larger and more sophisticated,

but the learning process remains exactly the same.

---

# 5-Minute Mental Checklist

Whenever you think about neural network training, remember this sequence:

```text
Input

↓

Forward Pass

↓

Prediction

↓

Loss Function

↓

Computational Graph

↓

Backpropagation

↓

Chain Rule

↓

Gradient Vector

↓

Gradient Descent

↓

Updated Weights

↓

Repeat
```

If you can visualize this entire flow, you understand the core training process used by modern deep learning models.

---

# Looking Ahead

So far, we've learned **how** a neural network learns.

One important question still remains.

> **Can we make learning faster, more stable, and more efficient?**

The answer lies in optimization techniques such as:

- Learning Rate
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent
- Momentum
- Adam

These will be the focus of the next part of our Deep Learning journey.