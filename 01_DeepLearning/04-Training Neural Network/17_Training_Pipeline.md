# Training Pipeline — Putting Everything Together

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- What happens during one training iteration?
- How do all the concepts we've learned fit together?
- What is a Forward Pass?
- What is a Backward Pass?
- When are gradients computed?
- When are weights updated?
- What is an iteration?
- What is an epoch?
- Why does the loss gradually decrease during training?

---

# Chapter 1 — Watching the Entire Movie

Over the last several lectures, we've learned many individual concepts.

- Loss Functions
- Cross Entropy
- Gradient Descent
- Derivatives
- Partial Derivatives
- Gradient Vector
- Backpropagation
- Chain Rule
- Computational Graph

Each concept made sense individually.

But imagine watching a movie by looking at only one frame at a time.

One frame shows the hero.

Another shows the villain.

Another shows the climax.

Although every frame is meaningful, you cannot understand the entire story until you watch the complete movie.

The same is true for neural networks.

So far, we've learned the individual building blocks.

Now it's time to see how they all work together.

This lecture tells the complete story of how a neural network learns.

---

# Chapter 2 — The Big Picture

Training a neural network is simply a loop.

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
Backpropagation
      │
      ▼
Gradient Computation
      │
      ▼
Optimizer
(Gradient Descent / Adam)
      │
      ▼
Updated Weights
      │
      ▼
Repeat...
```

Every training iteration follows exactly the same sequence.

---

# Chapter 3 — Step 1: Forward Pass

Everything begins with an input.

Suppose our task is image classification.

```text
Image
   │
   ▼
Neural Network
```

The image passes through multiple layers.

```text
Input
   │
   ▼
Linear Layer
(z₁ = W₁x + b₁)
   │
   ▼
ReLU
(a₁ = ReLU(z₁))
   │
   ▼
Linear Layer
(z₂ = W₂a₁ + b₂)
   │
   ▼
Softmax
(Probability Distribution)
```

Eventually, the network predicts

```text
Dog      = 2%

Cat      = 97%

Horse    = 1%
```

This is called the **Forward Pass** because information flows from the input layer to the output layer.

---

## Deep Dive — From Perceptron to Modern Neural Networks

You might notice that modern neural networks look different from the Perceptron we studied earlier.

The original Perceptron looked like this:

```text
Input
   │
   ▼
Weighted Sum
(z = Wx + b)
   │
   ▼
Step Function
   │
   ▼
Output (0 or 1)
```

Modern neural networks no longer use the Step Function.

Instead, they separate the computation into two stages.

```text
Input
   │
   ▼
Linear Layer
(z = Wx + b)
   │
   ▼
Activation Function
(ReLU / GELU / Sigmoid / Tanh)
   │
   ▼
Next Layer
```

Notice something important.

A **Linear Layer** performs only the weighted sum

$$
z = Wx + b
$$

The activation function is applied **after** the Linear Layer.

For example,

```text
Linear Layer

↓

-5

↓

ReLU

↓

0
```

or

```text
Linear Layer

↓

8

↓

ReLU

↓

8
```

Unlike the Step Function, ReLU receives the raw weighted sum (`z`) and preserves positive values instead of converting everything into only `0` or `1`.

This separation allows deep neural networks to learn complex non-linear patterns while remaining trainable using Backpropagation.

---

# Chapter 4 — Step 2: Compute the Loss

The model predicts

```text
Cat = 97%
```

Now compare it with the ground truth.

```text
True Label = Cat
```

The Loss Function measures how good or bad the prediction is.

Suppose Cross Entropy computes

```text
Loss = 0.08
```

A small loss means

the prediction was good.

A large loss means

the prediction was poor.

---

# Chapter 5 — Step 3: Build the Computational Graph

While making the prediction, the framework is secretly recording every operation.

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

It remembers how every value depends on the previous one.

The graph will soon be used during Backpropagation.

---

# Chapter 6 — Step 4: Backpropagation

Now the learning begins.

Starting from the loss,

the framework moves backward through the graph.

```text
Loss
   ▲
   │
Softmax
   ▲
   │
Linear Layer
   ▲
   │
ReLU
   ▲
   │
Linear Layer
```

At every step,

the Chain Rule computes

how much each weight contributed to the final loss.

Eventually,

every trainable parameter receives its gradient.

```text
∂L/∂w₁

∂L/∂w₂

∂L/∂w₃

...
```

---

# Chapter 7 — Step 5: Update the Weights

Now that every gradient is known,

the optimizer updates the weights.

In this module, the optimizer is **Gradient Descent**.

$$
w = w - \eta \nabla L
$$

Later in this course, we'll replace Gradient Descent with more advanced optimizers such as **Momentum** and **Adam**.

After the update,

the neural network has learned something from this training example.

---

# Chapter 8 — Step 6: Repeat

Training does not stop after one training iteration.

The next batch of training data arrives.

Again,

```text
Forward Pass

↓

Loss

↓

Backpropagation

↓

Weight Update
```

This process repeats

thousands,

millions,

or even billions of times.

Gradually,

the weights improve.

The predictions become better.

The loss becomes smaller.

---

# Chapter 9 — Iterations vs Epochs

These two terms are often confused.

A single cycle of

```text
Forward Pass

↓

Loss

↓

Backpropagation

↓

Weight Update
```

is called **one training iteration**.

Suppose our dataset contains

```text
10,000 images
```

One complete pass through all 10,000 images is called

> **One Epoch**

If the model trains for

```text
20 Epochs
```

it has seen every training example twenty times.

Each epoch usually consists of many training iterations.

---

# Chapter 10 — The Entire Learning Process

Everything we've studied can now be summarized in one diagram.

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
Optimizer
(Gradient Descent / Adam)
       │
       ▼
Updated Weights
       │
       ▼
Next Iteration
```

This is the complete learning cycle of a neural network.

---

# Chapter 11 — Connecting This to GPT

Exactly the same pipeline is used to train GPT.

Suppose the sentence is

```text
I love deep ______
```

The correct next token is

```text
learning
```

The Forward Pass predicts

```text
learning = 72%

AI = 18%

math = 5%

science = 5%
```

Cross Entropy computes the loss.

Backpropagation computes gradients.

The optimizer updates billions of weights.

The next time the same pattern appears,

the probability of **learning** becomes slightly higher.

Repeating this process across trillions of tokens is how GPT gradually learns language.

---

# Interview Insight

A common interview question is:

> **Explain the complete training pipeline of a neural network.**

A strong answer is:

1. Perform a Forward Pass to generate predictions.
2. Compute the Loss by comparing predictions with the ground truth.
3. Build the Computational Graph during the forward pass.
4. Apply Backpropagation using the Chain Rule to compute gradients.
5. Use an optimizer (such as Gradient Descent or Adam) to update the weights.
6. Repeat this process for every batch across multiple epochs until the model converges.

---

# Key Takeaways

- Training is an iterative process.
- The Forward Pass computes predictions.
- The Loss Function measures prediction quality.
- The Computational Graph records every operation.
- Backpropagation computes gradients using the Chain Rule.
- The optimizer updates the model parameters.
- Repeating this process over many epochs gradually improves the model.