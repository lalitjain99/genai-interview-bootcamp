# Negative Log Likelihood — The Secret Behind Cross Entropy

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- What is Negative Log Likelihood (NLL)?
- Why does PyTorch provide `NLLLoss` if Cross Entropy already exists?
- What does "Likelihood" actually mean?
- Why is the logarithm used again?
- How is NLL related to Cross Entropy?
- Why does `CrossEntropyLoss` internally use `LogSoftmax` and `NLLLoss`?

---

# Chapter 1 — A Curious Discovery

So far, we've learned about two classification loss functions.

For multi-class classification:

```text
Softmax
        ↓
Cross Entropy
```

For binary classification:

```text
Sigmoid
        ↓
Binary Cross Entropy
```

Everything seems complete.

Then you open the PyTorch documentation and discover another loss function.

```python
nn.NLLLoss()
```

Immediately a question appears.

> "Why do we need another loss function?"

Doesn't Cross Entropy already solve the problem?

To answer this question, we first need to understand the meaning of one important word.

**Likelihood.**

---

# Chapter 2 — What Does "Likelihood" Mean?

Suppose a neural network predicts the following probabilities.

```text
Dog      = 2%

Cat      = 97%

Horse    = 1%
```

The correct answer is

```text
Cat
```

Notice what we're really interested in.

We don't actually care about all three probabilities.

We only care about the probability assigned to the **correct class**.

In this case,

```text
Likelihood = 97%
```

Now imagine another model.

```text
Dog      = 30%

Cat      = 45%

Horse    = 25%
```

The correct class is still

```text
Cat
```

Now the likelihood becomes

```text
45%
```

The second model assigned a much lower probability to the correct answer.

Therefore,

its prediction is worse.

---

# Chapter 3 — The Goal of Training

Think like an engineer.

During training,

what do we actually want?

We want the model to assign

**higher and higher probability**

to the correct answer.

In other words,

we want to maximize the likelihood.

```text
Likelihood ↑

Model becomes better.
```

This is why statisticians called it

**Maximum Likelihood Estimation (MLE).**

Training tries to maximize the probability of observing the correct answer.

---

# Chapter 4 — But Gradient Descent Minimizes

Here comes the problem.

Our optimizer is Gradient Descent.

Gradient Descent always minimizes a function.

```text
Loss ↓
```

But likelihood is something we want to maximize.

```text
Likelihood ↑
```

The objectives are opposite.

So how do we convert a maximization problem into a minimization problem?

Simple.

Multiply it by **−1**.

```text
Maximize

Likelihood

↓

Minimize

−Likelihood
```

Now Gradient Descent can optimize it.

---

# Chapter 5 — Why Take the Log?

Earlier, we already learned why Cross Entropy uses logarithms.

The log has two useful properties.

1. It strongly penalizes confident mistakes.

2. It converts products into sums.

Instead of minimizing

```text
−Likelihood
```

we minimize

```text
−log(Likelihood)
```

This quantity is called

**Negative Log Likelihood (NLL).**

---

# Chapter 6 — A Simple Example

Suppose the correct class is

```text
Cat
```

Prediction A

```text
P(Cat) = 0.99
```

Loss

```text
−log(0.99)

≈ 0.01
```

Almost zero.

Excellent prediction.

---

Prediction B

```text
P(Cat) = 0.50
```

Loss

```text
−log(0.50)

≈ 0.69
```

Larger penalty.

---

Prediction C

```text
P(Cat) = 0.01
```

Loss

```text
−log(0.01)

≈ 4.61
```

Huge penalty.

Exactly the behavior we wanted.

---

# Chapter 7 — Wait... Isn't This Cross Entropy?

At this point,

you may notice something surprising.

The calculation looks familiar.

Cross Entropy also computes

```text
−log(probability of the correct class)
```

So what's the difference?

The answer is:

**There is almost no difference.**

Cross Entropy is essentially

```text
LogSoftmax

↓

Negative Log Likelihood
```

combined into one operation.

---

# Chapter 8 — Inside PyTorch

Suppose your model produces logits.

```text
2.1

5.4

1.3
```

If you use

```python
nn.CrossEntropyLoss()
```

PyTorch internally performs

```text
Logits

↓

LogSoftmax

↓

Negative Log Likelihood
```

You never see these intermediate steps.

Everything happens automatically.

---

If you instead use

```python
nn.NLLLoss()
```

then **you** must apply

```python
log_softmax()
```

before passing the values to the loss function.

```python
output = F.log_softmax(logits, dim=1)
loss = nn.NLLLoss()(output, target)
```

---

# Chapter 9 — A Common Confusion: Is LogSoftmax Equal to `-log(p)`?

One of the most common misconceptions is thinking that **LogSoftmax** computes

$$
-\log(p)
$$

This is **not** true.

Let's carefully separate the two operations.

---

## Step 1 — The Neural Network Produces Logits

Suppose the final layer outputs

```text
Dog      = 2.0

Cat      = 5.0

Horse    = 1.0
```

These values are called **logits**.

They are not probabilities.

---

## Step 2 — Softmax Converts Logits into Probabilities

After applying Softmax, we obtain

```text
Dog      = 0.046

Cat      = 0.936

Horse    = 0.018
```

Now every value is a valid probability.

---

## Step 3 — LogSoftmax Takes the Logarithm

LogSoftmax simply computes the logarithm of every probability.

```text
Dog      = log(0.046)  = -3.08

Cat      = log(0.936)  = -0.066

Horse    = log(0.018)  = -4.02
```

Notice something important.

LogSoftmax **does not apply the negative sign**.

It only computes

$$
\log(p)
$$

for every class.

Since probabilities always lie between 0 and 1,

their logarithms are always less than or equal to zero.

---

## Step 4 — NLLLoss Applies the Negative Sign

Suppose the correct class is

```text
Cat
```

NLLLoss ignores all the other values.

It selects only the log probability of the correct class.

```text
Cat = -0.066
```

Then it simply multiplies it by **−1**.

```text
Loss

=

-(-0.066)

=

0.066
```

This is where

$$
-\log(p)
$$

comes from.

The negative sign is applied by **Negative Log Likelihood**, not by LogSoftmax.

---

## Putting Everything Together

The complete pipeline looks like this.

```text
Neural Network

        │
        ▼

      Logits

        │
        ▼

     Softmax

        │
        ▼

   Probabilities

        │
        ▼

   LogSoftmax

        │
        ▼

 Log Probabilities

        │
        ▼

     NLLLoss

        │
        ▼

  −log(correct probability)
```

---

## Why Does `CrossEntropyLoss` Feel Simpler?

PyTorch combines these two operations into one.

Instead of writing

```python
log_probs = F.log_softmax(logits, dim=1)
loss = nn.NLLLoss()(log_probs, target)
```

we simply write

```python
loss = nn.CrossEntropyLoss()(logits, target)
```

Internally, PyTorch performs exactly the same sequence of operations.

```text
Logits

↓

LogSoftmax

↓

NLLLoss

↓

Final Loss
```

This is why we often say

> **CrossEntropyLoss = LogSoftmax + NLLLoss**

---

## Interview Insight

A common interview question is:

> **Does LogSoftmax compute `-log(p)`?**

**Answer:**

No.

- **LogSoftmax** computes **`log(p)`** for every class.
- **NLLLoss** selects the correct class and applies the negative sign.
- Together, they produce **`-log(p_correct)`**, which becomes the final loss.

---

# Chapter 10 — Cross Entropy vs NLL

| Cross Entropy | Negative Log Likelihood |
|---------------|--------------------------|
| Accepts logits | Accepts log probabilities |
| Applies LogSoftmax internally | Expects LogSoftmax already applied |
| Easier to use | Gives more flexibility |
| Most commonly used | Mostly used internally or in advanced models |

---

# Chapter 11 — Connecting This to GPT

GPT predicts logits for every token in its vocabulary.

```text
Logits

↓

CrossEntropyLoss

↓

Training Loss
```

Internally,

PyTorch performs

```text
Logits

↓

LogSoftmax

↓

Negative Log Likelihood
```

Even though you never explicitly call `NLLLoss`,

it is still part of the training pipeline.

---

# Looking Ahead

So far, we've answered an important question.

We know **what** loss function to minimize.

But another mystery remains.

How does the neural network know

**which weight**

should change?

How does it know

whether to increase a weight,

decrease it,

or leave it unchanged?

To answer that,

we must first understand one of the most important ideas in all of machine learning.

**Derivatives.**

---

# Key Takeaways

- Likelihood is the probability assigned to the correct class.
- Training aims to maximize likelihood.
- Gradient Descent minimizes, so we minimize the negative likelihood instead.
- Taking the logarithm gives us Negative Log Likelihood.
- `CrossEntropyLoss` is essentially `LogSoftmax + NLLLoss`.
- NLLLoss expects log probabilities as input, while CrossEntropyLoss expects raw logits.