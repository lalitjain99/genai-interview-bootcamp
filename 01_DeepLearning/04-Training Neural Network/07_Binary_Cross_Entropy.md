# Binary Cross Entropy — When There Are Only Two Classes

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- Why isn't ordinary Cross Entropy always used?
- What is Binary Cross Entropy (BCE)?
- Why is BCE designed specifically for binary classification?
- Why is Sigmoid almost always paired with BCE?
- Can Softmax also solve binary classification?
- Why does PyTorch provide both `BCELoss` and `BCEWithLogitsLoss`?
- When should we use BCE instead of Cross Entropy?

---

# Chapter 1 — A New Problem Appears

So far, we've studied **multi-class classification**.

For example, consider an image classification problem.

```text
Dog
Cat
Horse
```

Exactly one of these classes is correct.

To solve this problem, the output layer contains **one neuron for each class**.

Softmax converts the logits produced by these neurons into probabilities.

```text
Dog     = 2%

Cat     = 97%

Horse   = 1%
```

Cross Entropy then measures how good these predicted probabilities are.

Everything works perfectly.

---

Now consider a different problem.

Suppose we are building a **spam detection system**.

For every email, there are only two possibilities.

```text
Spam

Not Spam
```

Unlike image classification, there isn't a third possible answer.

This naturally raises an interesting question.

> **Do we still need multiple output neurons and Softmax?**

---

# Chapter 2 — Why One Output Neuron Is Enough

Let's think about what these two classes represent.

```text
Spam

Not Spam
```

These are exact opposites.

If the model becomes **more confident** that an email is spam,

it automatically becomes **less confident** that it is not spam.

Mathematically,

```text
P(Spam) + P(Not Spam) = 1
```

Suppose

```text
P(Spam) = 0.92
```

Then,

```text
P(Not Spam) = 1 − 0.92 = 0.08
```

Notice something interesting.

Once we know the probability of one class,

the probability of the other class is automatically determined.

Unlike multi-class classification,

we don't need multiple neurons competing with each other.

A single output neuron is sufficient.

---

# Chapter 3 — From Logit to Probability

Suppose our single output neuron produces the following value.

```text
4.5
```

This value is called a **logit**.

Just like before,

a logit is **not** a probability.

We therefore need a function that converts this value into a probability.

Instead of Softmax,

we apply the **Sigmoid** function.

The Sigmoid function always produces values between **0** and **1**.

For example,

```text
Logit = 4.5
```

After applying Sigmoid,

```text
P(Spam) = 0.989
```

The probability of **Not Spam** becomes

```text
P(Not Spam) = 1 − 0.989 = 0.011
```

Notice that we never explicitly computed the second probability.

It came automatically.

---

# Chapter 4 — Why Cross Entropy Needs a Binary Version

Suppose the true label is

```text
Spam
```

The ground truth can therefore be represented as

```text
y = 1
```

Now consider three predictions.

### Prediction 1

```text
P(Spam) = 0.99
```

Excellent prediction.

The loss should be very small.

---

### Prediction 2

```text
P(Spam) = 0.60
```

Correct,

but not very confident.

The loss should be larger.

---

### Prediction 3

```text
P(Spam) = 0.02
```

The model is completely wrong.

Even worse,

it was extremely confident.

This prediction deserves a very large penalty.

Notice that this is exactly the behavior we wanted from Cross Entropy.

The only difference is that we now have **one probability** instead of many.

This is why Binary Cross Entropy exists.

It is simply a version of Cross Entropy designed specifically for binary classification.

---

# Chapter 5 — Multi-Class vs Binary Classification

| Multi-Class Classification | Binary Classification |
|-----------------------------|------------------------|
| Three or more classes | Exactly two classes |
| One output neuron per class | One output neuron |
| Softmax activation | Sigmoid activation |
| Cross Entropy Loss | Binary Cross Entropy Loss |

Notice the beautiful symmetry.

```text
Softmax
        ↓
Cross Entropy

Sigmoid
        ↓
Binary Cross Entropy
```

---

# Chapter 6 — Can We Still Use Softmax?

A common interview question is:

> **Can Softmax also solve binary classification?**

Yes.

We could build an output layer containing two neurons.

```text
Spam

Not Spam
```

Softmax might produce

```text
Spam        = 98%

Not Spam    = 2%
```

This approach is perfectly valid.

However,

it uses two output neurons when only one is actually needed.

Most modern binary classifiers therefore use

```text
One Output Neuron

↓

Sigmoid

↓

Binary Cross Entropy
```

This architecture is simpler,

requires fewer parameters,

and is computationally more efficient.

---

# Chapter 7 — Real-World Applications

### Binary Classification

```text
Spam Detection

Fraud Detection

Disease Prediction

Loan Approval

Customer Churn Prediction
```

Typically uses

```text
Sigmoid

+

Binary Cross Entropy
```

---

### Multi-Class Classification

```text
Image Classification

Language Identification

Digit Recognition

Animal Classification
```

Typically uses

```text
Softmax

+

Cross Entropy
```

---

### GPT

GPT predicts the next token from a vocabulary containing tens of thousands of possible words.

Each token acts as a separate class.

Therefore GPT uses

```text
Softmax

+

Cross Entropy
```

—not Binary Cross Entropy.

---

# Looking Ahead

We've now completed the family of classification loss functions.

We understand:

- Cross Entropy for multiple classes.
- Binary Cross Entropy for binary classification.

But another question remains.

Deep learning libraries like PyTorch also provide

```python
nn.NLLLoss()
```

If Cross Entropy already works so well,

why does another loss function exist?

Is it solving a different problem,

or is it secretly connected to Cross Entropy?

That is exactly what we'll uncover in the next lecture.

---

# Key Takeaways

- Binary Cross Entropy is designed specifically for binary classification.
- Binary classification usually requires only one output neuron.
- Sigmoid converts the output logit into a probability.
- Binary Cross Entropy measures how good that probability is.
- Softmax can also solve binary classification but is usually unnecessary.
- In practice, **Sigmoid + Binary Cross Entropy** is the standard combination for binary classification.