# Perceptron — The Birth of Neural Networks

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- Why wasn't traditional Machine Learning enough?
- What is Feature Engineering?
- Why did Feature Engineering become the bottleneck?
- What is Representation Learning?
- What is a Perceptron?
- How does a Perceptron make predictions?
- Why do we need Weights and Bias?
- Why is the Perceptron still a linear model?
- Why can't it solve XOR?
- Why did Deep Learning continue evolving beyond the Perceptron?

---

# Chapter 1 — The Problem Before Deep Learning

Before Deep Learning,

Machine Learning looked like this.

```text
Raw Data
      ↓
Human Feature Engineering
      ↓
Machine Learning Model
      ↓
Prediction
```

The model itself was actually **not** the biggest challenge.

The real bottleneck was something else.

Humans had to decide

> Which information should be given to the model?

This process is called

> **Feature Engineering**

---

# Chapter 2 — Why Feature Engineering Became the Bottleneck

Imagine building a Face Recognition system.

Before training the model,

an engineer had to manually decide features such as

- Eye distance
- Nose width
- Face height
- Skin tone
- Jaw angle

Only after these handcrafted features were created

could algorithms like

- Linear Regression
- Logistic Regression
- Decision Trees
- SVM

learn from them.

The machine never discovered useful features by itself.

Humans had to tell it what was important.

As problems became more complex,

this became impossible.

---

# Chapter 3 — The Big Question

Researchers started asking

> Can a machine learn useful features automatically?

Instead of

```text
Human
↓

Features
↓

Model
```

could we build

```text
Raw Data
↓

Neural Network
↓

Useful Features
↓

Prediction
```

This idea completely changed Artificial Intelligence.

It is called

> **Representation Learning**

Instead of learning only the final prediction,

the network first learns how to represent the data.

---

# Chapter 4 — The First Neural Network

The simplest neural network ever proposed was

> **The Perceptron**

It receives inputs

```text
x₁

x₂

...

xₙ
```

Each input has an associated weight.

The weighted inputs are added together.

Finally,

a bias is added.

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

or

```text
z = Wx + b
```

This value is then passed through an activation function.

---

# Chapter 5 — Why Do We Need Weights?

Suppose we are predicting whether a person receives a loan.

Features are

```text
Salary

Credit Score

Age
```

Should all features contribute equally?

Clearly not.

Salary is usually much more important than Age.

Weights allow the model to decide

- which feature is important,
- how important it is,
- and whether it increases or decreases the prediction.

Positive weight

```text
Feature ↑

Prediction ↑
```

Negative weight

```text
Feature ↑

Prediction ↓
```

Therefore,

weights learn both

- direction
- magnitude

of influence.

---

# Chapter 6 — Why Do We Need Bias?

Suppose there are no bias terms.

The equation becomes

```text
z = Wx
```

Every decision boundary must now pass through

```text
Origin (0,0)
```

Many simple classification problems become impossible.

Bias allows us to shift the decision boundary.

Instead of

```text
Always here
```

we can move it anywhere in the feature space.

Think of Bias as

> Moving the entire decision boundary without changing its orientation.

---

# Chapter 7 — Activation Function

After computing

```text
z = Wx + b
```

the Perceptron applies the Step Function.

```text
if z > 0

Output = 1

else

Output = 0
```

Instead of producing a continuous value,

the model now makes a binary decision.

---

# Chapter 8 — Is the Perceptron Really Non-Linear?

Many beginners believe

> "The Step Function is non-linear, so the Perceptron must be non-linear."

This is incorrect.

The weighted sum

```text
Wx + b
```

still creates only

```text
One Hyperplane
```

The Step Function merely decides

which side of that hyperplane the input belongs to.

The decision boundary remains linear.

---

# Chapter 9 — The Biggest Limitation

Because there is only one decision boundary,

the Perceptron cannot solve problems like

- XOR
- Circular decision boundaries
- Images
- Speech
- Natural Language

All these problems require

multiple non-linear transformations.

One linear boundary is simply not enough.

This limitation eventually led researchers toward

Hidden Layers

and later

Deep Neural Networks.

---

# Engineer's Insight

The Perceptron introduced three ideas that still exist inside every modern neural network.

```text
Input

↓

Weights

↓

Bias

↓

Activation

↓

Output
```

Even GPT follows this same philosophy.

The architecture became deeper.

The activation function became differentiable.

The number of parameters became billions.

But the fundamental computation remains remarkably similar.

# First-Principles Chain

```text
Why wasn't traditional ML enough?
        │
        ▼
Feature Engineering became the bottleneck.
        │
        ▼
Can machines learn features automatically?
        │
        ▼
Representation Learning
        │
        ▼
What is the simplest model that can learn?
        │
        ▼
Perceptron
        │
        ▼
Can one Perceptron solve every problem?
        │
        ▼
No.
        │
        ▼
Why?
        │
        ▼
It is still a linear classifier.
        │
        ▼
How do we solve that?
        │
        ▼
Hidden Layers

---

# Key Takeaways

- Traditional Machine Learning depended on manual Feature Engineering.
- Deep Learning introduced Representation Learning.
- A Perceptron is the simplest neural network.
- Weights determine the direction and magnitude of each feature's influence.
- Bias shifts the decision boundary away from the origin.
- The Step Function converts a continuous value into a binary prediction.
- Despite using a non-linear activation, the Perceptron remains a linear classifier.
- It cannot solve non-linearly separable problems such as XOR.
- The limitations of the Perceptron motivated Hidden Layers and modern Deep Learning.

---

# Looking Ahead

The obvious question now is:

> **If one Perceptron is still linear, what happens if we stack multiple Perceptrons together?**

Can multiple neurons solve XOR?

Can they learn complex patterns?

That is exactly what we'll explore in the next lecture on **Hidden Layers & XOR**.