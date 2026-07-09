# Computational Graph — How PyTorch Automatically Computes Gradients

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- What is a Computational Graph?
- Why do we need a Computational Graph?
- How does a neural network become a graph?
- How does PyTorch compute gradients automatically?
- What is Automatic Differentiation (Autograd)?
- How are Computational Graphs connected to the Chain Rule?

---

# Chapter 1 — A New Problem

In the previous lecture, we learned that the Chain Rule allows us to compute the effect of a weight on the final loss.

For a simple network, we can manually write

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\times
\frac{\partial a}{\partial z}
\times
\frac{\partial z}{\partial w}
$$

This isn't too difficult.

But now imagine GPT.

Instead of three mathematical operations,

it performs **millions of operations**.

Imagine writing the Chain Rule manually for every operation.

Clearly,

that is impossible.

A natural question arises.

> **How does PyTorch know which derivatives to multiply together?**

---

# Chapter 2 — Think Like Google Maps

Imagine you're using Google Maps.

The application shows a network of roads.

```text
Home

↓

Road A

↓

Road B

↓

Office
```

Google Maps doesn't remember only the starting point and destination.

It remembers **every road connecting them**.

Why?

Because if one road becomes blocked,

it knows exactly how every location is connected.

A neural network behaves in a similar way.

Instead of roads,

it stores mathematical operations.

---

# Chapter 3 — Every Computation Becomes a Node

Consider a simple equation.

$$
z = wx + b
$$

Instead of viewing it as one equation,

a framework breaks it into smaller operations.

```text
      w        x
       \      /
        Multiply
            │
            ▼
          (+ b)
            │
            ▼
            z
```

Notice something.

Every mathematical operation becomes a node.

The graph stores

- the inputs,
- the operation performed,
- and the resulting output.

This is called a **Computational Graph**.

---

# Chapter 4 — A Neural Network Is Just a Bigger Graph

Now imagine a neural network.

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

A deep learning framework converts this into a graph of operations.

```text
Input

↓

Multiply

↓

Add Bias

↓

ReLU

↓

Multiply

↓

Add Bias

↓

Softmax

↓

Cross Entropy

↓

Loss
```

The graph records every operation that happened during the forward pass.

---

# Chapter 5 — The Forward Pass Builds the Graph

One surprising fact about PyTorch is this.

During the forward pass,

the framework is doing two things at the same time.

First,

it computes the prediction.

Second,

it secretly records every operation.

For example,

when you write

```python
z = w * x + b
```

PyTorch computes the value of **z**,

but it also remembers

```text
z

came from

Multiply

followed by

Addition
```

By the end of the forward pass,

the entire computational graph has been built.

---

# Chapter 6 — Backpropagation Walks Back Through the Graph

Suppose the graph looks like this.

```text
Input

↓

Multiply

↓

ReLU

↓

Softmax

↓

Loss
```

During Backpropagation,

PyTorch starts at the loss.

Then it moves backward.

```text
Loss

↑

Softmax

↑

ReLU

↑

Multiply

↑

Input
```

At every node,

PyTorch applies the **local derivative**.

This is nothing more than the Chain Rule being applied repeatedly.

---

# Chapter 7 — Automatic Differentiation (Autograd)

You might wonder,

> **Do we have to write all these derivatives ourselves?**

Fortunately,

no.

PyTorch includes a system called **Autograd**.

Autograd automatically

- builds the computational graph,
- applies the Chain Rule,
- computes every gradient,
- stores the gradients inside each parameter.

This is why training a neural network can be as simple as

```python
loss.backward()
```

Behind that single line,

PyTorch performs thousands—or even millions—of derivative calculations.

---

# Chapter 8 — Connecting Everything

Let's connect all the ideas we've learned.

```text
Input
      │
      ▼
Forward Pass
      │
      ▼
Computational Graph
      │
      ▼
Prediction
      │
      ▼
Loss
      │
      ▼
Backpropagation
      │
      ▼
Chain Rule
      │
      ▼
Gradients
      │
      ▼
Gradient Descent
      │
      ▼
Updated Weights
```

Notice how every topic we've studied fits into one complete learning pipeline.

---

# Chapter 9 — Looking Ahead

So far,

we've understood

- Loss Functions
- Gradient Descent
- Derivatives
- Partial Derivatives
- Gradient Vector
- Backpropagation
- Chain Rule
- Computational Graph

Only one question remains.

> **What does the complete training process look like from start to finish?**

How does a training example move through

- the Forward Pass,
- the Loss Function,
- Backpropagation,
- Gradient Descent,

and finally produce an updated model?

That complete picture is the **Training Pipeline**,

which we'll study in the next lecture.

---

# Key Takeaways

- A Computational Graph represents every mathematical operation performed during the forward pass.
- Every operation becomes a node in the graph.
- The forward pass both computes outputs and builds the graph.
- Backpropagation traverses the graph in reverse order.
- At each node, the Chain Rule is applied locally.
- PyTorch's Autograd automatically builds the graph and computes gradients.
- Calling `loss.backward()` triggers Backpropagation over the Computational Graph.

---

# Interview Insight

A common interview question is:

> **What is the relationship between Backpropagation, the Chain Rule, and the Computational Graph?**

The answer is:

- **The Computational Graph** stores all mathematical operations performed during the forward pass.
- **The Chain Rule** provides the mathematical rule for propagating gradients through those operations.
- **Backpropagation** is the algorithm that traverses the Computational Graph in reverse, repeatedly applying the Chain Rule to compute gradients efficiently.