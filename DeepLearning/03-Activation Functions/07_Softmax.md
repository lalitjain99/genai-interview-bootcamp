# Softmax — When Neurons Start Competing

## Learning Objectives

By the end of this lecture, you should be able to answer:

- Why can't ReLU or GELU produce probabilities?
- What are logits?
- Why does the output layer have one neuron per class?
- Why do we need Softmax?
- Why do Softmax probabilities always sum to one?
- Why does every modern classifier and GPT end with Softmax?

---

# Chapter 1 — The Problem

Imagine you've built an image classifier.

After many hidden layers, the network produces the following output.

```text
Dog    = 4.2

Cat    = 7.8

Horse  = 2.5
```

Question:

Can we already say the image is a **Cat**?

Not really.

These numbers tell us that the network prefers **Cat** over **Dog** and **Horse**.

But they don't tell us **how confident** the model is.

Does **7.8** mean:

- 60% confidence?
- 80% confidence?
- 99% confidence?

There is no way to know.

These numbers are **scores**, not probabilities.

Modern AI systems don't just make predictions.

They also estimate **how confident** they are.

This is exactly the problem Softmax was designed to solve.

---

# Chapter 2 — What Are Logits?

The numbers

```text
Dog    = 4.2

Cat    = 7.8

Horse  = 2.5
```

are called **logits**.

A logit is simply the raw output produced by the final linear layer of a neural network.

Mathematically,

$$
z = Wx + b
$$

Notice something beautiful.

This equation should look very familiar.

It is exactly the same weighted sum that we first learned while studying the Perceptron.

Even GPT's final layer still computes

$$
z = Wx + b
$$

The only difference is that today's neural networks learn much richer feature representations before reaching this final layer.

Logits are simply these final scores.

They can be positive.

They can be negative.

They can be greater than one.

They are **not probabilities**.

---

# Chapter 3 — Why ReLU or GELU Cannot Help

You might wonder,

> Why not simply apply ReLU?

Suppose the logits are

```text
Dog    = 4

Cat    = 8

Horse  = 2
```

After ReLU,

```text
Dog    = 4

Cat    = 8

Horse  = 2
```

Nothing changes.

Still not probabilities.

Now suppose we use GELU.

```text
Dog    = 3.98

Cat    = 7.99

Horse  = 1.95
```

Again,

- values don't sum to one,
- values can still be greater than one,
- they cannot be interpreted as probabilities.

Why?

Because ReLU and GELU treat every neuron **independently**.

Classification is different.

Every class must compete against every other class.

Softmax is the first activation function we've studied that operates on the **entire output vector** rather than a single neuron.

---

# Chapter 4 — How the Output Layer Works

At this point, a natural question appears.

> Why does the network produce multiple outputs for a single image?

Suppose we want to classify three animals.

- Dog
- Cat
- Horse

The output layer contains one neuron for each class.

```text
              Output Layer

          ○      ○      ○
         Dog    Cat   Horse
```

Each neuron learns to answer one question.

- "How likely is this a Dog?"
- "How likely is this a Cat?"
- "How likely is this a Horse?"

After the final linear layer, the neurons produce

```text
Dog    = 4.2

Cat    = 7.8

Horse  = 2.5
```

These are the logits.

---

## Why Not One Output Neuron?

Imagine the network had only one output neuron.

```text
7.8
```

Question:

What does **7.8** represent?

Dog?

Cat?

Horse?

Impossible to know.

Instead,

every class receives its own output neuron.

Each neuron specializes in scoring one possible class.

---

# Chapter 5 — Softmax Introduces Competition

Before Softmax,

```text
Dog    = 4.2

Cat    = 7.8

Horse  = 2.5
```

Question.

Is Cat twice as likely as Dog?

Ten times?

One hundred times?

The logits themselves don't answer this.

Softmax compares **all logits together**.

Instead of asking

> "How large is this score?"

it asks

> "How large is this score compared to every other score?"

The result becomes

```text
Dog    = 0.026

Cat    = 0.970

Horse  = 0.004
```

Now the outputs become

```text
2.6%

97.0%

0.4%
```

Notice something remarkable.

They always add up to

```text
100%
```

This is exactly what we expect from probabilities.

Unlike ReLU or GELU,

Softmax does not activate neurons independently.

It makes them compete.

Increasing the probability of one class automatically decreases the probabilities of the others.

---

# Chapter 6 — Connecting This to GPT

Exactly the same idea is used inside GPT.

Suppose the vocabulary contains

```text
50,000 tokens
```

The output layer therefore contains

```text
50,000 neurons
```

Each neuron represents one possible next token.

For example,

```text
apple      = 8.2

banana     = 2.1

orange     = 4.5

car        = 0.3

hello      = 5.9

...
```

These are logits.

Softmax converts them into

```text
P(apple)

P(banana)

P(orange)

...

Total = 1
```

The decoding algorithm then selects the next token.

---

# Interview Insight

**Does the number of output neurons always equal the number of classes?**

It depends on the task.

### Binary Classification

Two common approaches exist.

- One output neuron + Sigmoid
- Two output neurons + Softmax

Both are correct.

---

### Multi-Class Classification

One neuron per class.

Softmax converts logits into probabilities.

---

### GPT

Every vocabulary token acts as one possible class.

Therefore,

```
Number of Output Neurons = Vocabulary Size
```

This is why GPT's output layer contains tens of thousands of neurons.

---

# Chapter 7 — Looking Ahead

So far we have answered three important questions.

✅ What are logits?

✅ Why does every class have its own neuron?

✅ Why can't ReLU or GELU produce probabilities?

One question still remains.

How do we mathematically transform

```text
Dog    = 4.2

Cat    = 7.8

Horse  = 2.5
```

into

```text
Dog    = 2.6%

Cat    = 97%

Horse  = 0.4%
```

The answer lies in the Softmax equation.

---

# Chapter 8 — Deriving the Softmax Equation

So far, we know that Softmax converts **logits** into **probabilities**.

The obvious question is:

> **How?**

Let's first think like mathematicians.

Suppose our network outputs three logits.

```text
Dog    = 4.2

Cat    = 7.8

Horse  = 2.5
```

We want to convert these numbers into probabilities.

What properties should those probabilities satisfy?

---

## Requirement 1 — Every Probability Must Be Positive

A probability can never be negative.

For example,

```text
Dog = -20%
```

is meaningless.

Therefore, whatever function we choose must always produce **positive numbers**.

---

## Requirement 2 — Larger Logits Should Get Larger Probabilities

Suppose

```text
Cat = 7.8

Dog = 4.2
```

Since Cat has the larger logit,

its probability should also be larger.

The ordering should never change.

---

## Requirement 3 — All Probabilities Must Sum to One

If our model predicts

```text
Dog

Cat

Horse
```

the probabilities should satisfy

```text
P(Dog) + P(Cat) + P(Horse) = 1
```

This is the definition of a probability distribution.

---

Now we have a clear goal.

We need a mathematical function that

- always produces positive values,
- preserves ordering,
- and can be normalized into probabilities.

Surprisingly,

the exponential function satisfies all these properties beautifully.

---

# Chapter 9 — Why Do We Use the Exponential Function?

At first glance, the exponential function seems like a strange choice.

Why not simply divide each logit by the total?

For example,

```text
Dog    = 2

Cat    = 5

Horse  = 1
```

Could we compute

```text
Dog    = 2 / (2+5+1)

Cat    = 5 / (2+5+1)

Horse  = 1 / (2+5+1)
```

Sometimes this appears to work.

But consider another example.

```text
Dog    = -2

Cat    = 4

Horse  = 1
```

Now the probabilities become

```text
Dog = -2 / 3
```

A **negative probability** is impossible.

Simple normalization immediately breaks down.

---

The exponential function avoids this problem.

Recall a property from mathematics.

For every real number,

```text
e^x > 0
```

Whether

```text
x = -100

x = 0

x = 500
```

the exponential is **always positive**.

This immediately satisfies our first requirement.

---

Another beautiful property is that exponential preserves ordering.

If

```text
7 > 4
```

then

```text
e⁷ > e⁴
```

The class with the largest logit still receives the largest probability.

---

Finally,

the exponential magnifies confidence.

Consider two logits.

```text
5

6
```

Their difference is only

```text
1
```

After exponentiation,

```text
e⁵ ≈ 148

e⁶ ≈ 403
```

The larger score now becomes much more dominant.

This helps the model express confidence more clearly.

---

# Chapter 10 — The Softmax Equation

Now we have everything we need.

First,

convert every logit into a positive number using the exponential.

```text
Dog      → e^(Dog Logit)

Cat      → e^(Cat Logit)

Horse    → e^(Horse Logit)
```

Next,

add all these exponentials together.

Finally,

divide each exponential by the total.

Mathematically,

$$
P(y=i)=\frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

where

- \(z_i\) is the logit of class \(i\)
- \(n\) is the number of classes

This single equation guarantees that

- every probability is positive,
- every probability lies between 0 and 1,
- all probabilities sum to exactly one.

---

# Chapter 11 — Numerical Example

Suppose our logits are

```text
Dog    = 2

Cat    = 5

Horse  = 1
```

### Step 1 — Apply Exponential

```text
e² ≈ 7.39

e⁵ ≈ 148.41

e¹ ≈ 2.72
```

---

### Step 2 — Compute the Sum

```text
7.39 + 148.41 + 2.72

= 158.52
```

---

### Step 3 — Normalize

```text
Dog

7.39 / 158.52

≈ 0.047
```

```text
Cat

148.41 / 158.52

≈ 0.936
```

```text
Horse

2.72 / 158.52

≈ 0.017
```

Final probabilities become

```text
Dog      = 4.7%

Cat      = 93.6%

Horse    = 1.7%
```

Notice

```text
4.7

+

93.6

+

1.7

=

100%
```

Exactly what we wanted.

---

# Chapter 12 — Numerical Stability

Consider very large logits.

```text
1000

1002

998
```

Softmax requires computing

```text
e¹⁰⁰⁰
```

Unfortunately,

this number is so large that computers cannot represent it accurately.

This causes **overflow**.

To avoid this,

every deep learning framework performs one small trick.

Instead of computing

```text
Softmax(z)
```

it computes

```text
Softmax(z - max(z))
```

Suppose

```text
1000

1002

998
```

Subtract the maximum logit.

```text
-2

0

-4
```

Now the exponentials become

```text
e⁻²

e⁰

e⁻⁴
```

These values are small and safe to compute.

Most importantly,

subtracting the same constant from every logit **does not change the final probabilities**.

This optimization is used internally by frameworks like PyTorch and TensorFlow.

---

# Interview Insight

**Why do deep learning libraries compute**

```text
Softmax(x - max(x))
```

**instead of**

```text
Softmax(x)
```

Because exponentials grow extremely quickly.

Large logits can overflow.

Subtracting the maximum logit keeps every exponential numerically stable while producing exactly the same probability distribution.

---

# Key Takeaways

- Softmax converts logits into probabilities.
- Every output neuron contributes to the final probability distribution.
- Exponential guarantees positive values.
- Normalization guarantees the probabilities sum to one.
- Larger logits receive larger probabilities.
- Modern deep learning frameworks use a numerically stable implementation by subtracting the maximum logit before exponentiation.

---

## Looking Ahead

Softmax has now answered one question:

> **What is the probability of each class?**

But another question immediately arises.

How do we determine whether these probabilities are **good or bad**?

How should the model be penalized when it predicts

```text
Cat = 97%
```

versus

```text
Cat = 2%
```

That is the purpose of **Cross Entropy**, which we'll study in the next lecture.