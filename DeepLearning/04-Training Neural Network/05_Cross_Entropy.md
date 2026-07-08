# Cross Entropy Loss — Measuring Confidence in Classification

## Module 04 — Training Neural Networks

## Learning Objectives

By the end of this lecture, you should be able to answer:

- Why can't MSE or MAE be used for most classification problems?
- What does Cross Entropy actually measure?
- Why does Cross Entropy care about confidence?
- How does Cross Entropy punish confident mistakes?
- Why is Cross Entropy the standard loss function for classification?
- Why do Softmax and Cross Entropy almost always appear together?
- How does GPT use Cross Entropy during training?

---

# Chapter 1 — A Different Kind of Problem

Until now, we've been solving **regression** problems.

Examples:

- Predict house prices
- Predict rainfall
- Predict temperature

In all these cases, the output is a number.

Measuring numerical error makes perfect sense.

But now imagine a different problem.

We want to classify an image.

Possible classes are:

```text
Dog

Cat

Horse
```

Suppose the correct answer is

```text
Cat
```

Can we calculate

```text
Cat - Dog
```

No.

Categories don't have numerical distances.

This changes everything.

---

# Chapter 2 — The Model Doesn't Just Predict a Class

Modern neural networks don't simply predict

```text
Cat
```

Instead, after Softmax, they predict probabilities.

For example,

```text
Dog     = 2%

Cat     = 97%

Horse   = 1%
```

This prediction tells us two things.

- Which class the model chose.
- How confident it was.

Confidence is extremely important.

A learning algorithm should reward confident correct predictions and punish confident wrong predictions.

---

# Chapter 3 — Three Students Take the Same Exam

Imagine three students answering the same question.

Correct answer:

```text
Cat
```

---

### Student A

```text
Dog     = 1%

Cat     = 98%

Horse   = 1%
```

Excellent.

Almost no penalty.

---

### Student B

```text
Dog     = 30%

Cat     = 45%

Horse   = 25%
```

Correct.

But uncertain.

Medium penalty.

---

### Student C

```text
Dog     = 97%

Cat     = 2%

Horse   = 1%
```

Wrong.

And extremely confident.

Huge penalty.

---

Should all three students receive the same score?

Of course not.

A good loss function must distinguish between these three situations.

---

# Chapter 4 — Why MSE Isn't Enough

Let's compare Student A and Student B.

Both predicted

```text
Cat
```

Both have

```text
Accuracy = 100%
```

But Student A is much more confident.

Accuracy cannot see this difference.

Now suppose we use MSE.

MSE measures numerical distance.

But probabilities represent confidence, not numerical quantities.

MSE can be used for probabilities, but it doesn't strongly encourage the model to become highly confident about the correct class.

Classification needs a loss function specifically designed for probability distributions.

That loss function is **Cross Entropy**.

---

# Chapter 5 — What Does Cross Entropy Measure?

Cross Entropy asks one simple question.

> **"How much probability did the model assign to the correct class?"**

If the answer is

```text
99%
```

Very small loss.

If the answer is

```text
60%
```

Moderate loss.

If the answer is

```text
2%
```

Very large loss.

Notice something interesting.

Cross Entropy doesn't care much about the wrong classes.

It mainly focuses on one thing:

> **Did the correct class receive a high probability?**

---

# Chapter 6 — Confidence Matters

Suppose the correct answer is

```text
Cat
```

Prediction A

```text
Cat = 99%
```

Tiny loss.

---

Prediction B

```text
Cat = 75%
```

Medium loss.

---

Prediction C

```text
Cat = 5%
```

Huge loss.

The lower the probability assigned to the correct class,

the larger the penalty.

Cross Entropy teaches the model:

> "Don't just be correct."

> "Be confidently correct."

---

# Chapter 7 — Softmax and Cross Entropy

Remember our Softmax lecture.

Softmax converts logits into probabilities.

```text
Logits

↓

Softmax

↓

Probabilities
```

Cross Entropy comes immediately afterwards.

```text
Probabilities

↓

Cross Entropy

↓

Loss
```

Think of them as teammates.

Softmax answers

> "What probabilities should we assign?"

Cross Entropy answers

> "How good are those probabilities?"

Neither one replaces the other.

They solve different problems.

---

# Chapter 8 — Why Cross Entropy Became the Standard

Cross Entropy became the preferred loss function because it naturally encourages three behaviors.

✅ Increase the probability of the correct class.

✅ Decrease the probability of incorrect classes.

✅ Penalize confident mistakes much more heavily than uncertain mistakes.

These properties make it ideal for classification.

This is why you'll find Cross Entropy in:

- Image Classification
- Object Detection
- Speech Recognition
- Machine Translation
- Large Language Models

---

# Chapter 9 — GPT Connection

Suppose GPT receives the prompt

```text
The capital of France is
```

The correct next token is

```text
Paris
```

GPT predicts

```text
Paris      = 4%

London     = 90%

Berlin     = 3%

Rome       = 3%
```

The prediction is not just wrong.

It is confidently wrong.

Cross Entropy produces a very large loss.

Gradient Descent and Backpropagation then update billions of parameters so that next time,

the probability of **Paris** increases while the probability of **London** decreases.

This process happens billions of times during GPT's training.

---

# Mental Model

```text
Logits

↓

Softmax

↓

Probability Distribution

↓

Cross Entropy

↓

Loss

↓

Gradient Descent

↓

Better Predictions
```

---

# Key Takeaways

✅ MSE and MAE are designed for regression.

✅ Cross Entropy is designed for classification.

✅ Cross Entropy evaluates probability distributions.

✅ It rewards confident correct predictions.

✅ It strongly penalizes confident wrong predictions.

✅ Softmax and Cross Entropy are almost always used together.

---

# Looking Ahead

So far, we've treated Cross Entropy like a black box.

We know what it does.

But one mystery remains.

Why is the formula written as

$$
-\log(p)
$$

instead of something simpler?

Where does the logarithm come from?

Why do confident mistakes produce such enormous penalties?

In the next lecture, we'll derive the Cross Entropy formula from first principles and understand why the logarithm is the perfect mathematical choice.