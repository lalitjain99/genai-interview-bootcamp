# Softmax and Cross Entropy — Teaching the Model What "Wrong" Means

## Learning Objectives

By the end of this lecture, you should be able to answer:

- Why isn't predicting the correct class enough?
- Why do probabilities matter?
- Why can't accuracy be used as a loss function?
- Why are Softmax and Cross Entropy almost always used together?
- Why are confident mistakes punished more heavily?
- Why does GPT use Cross Entropy during training?

---

# Chapter 1 — The Teacher's Dilemma

Imagine you're a teacher checking multiple-choice exams.

Every student answered the same question.

> **What animal is shown in the picture?**

The correct answer is:

```text
Cat
```

Now let's look at three different students.

---

## Student A

```text
Dog      = 1%

Cat      = 98%

Horse    = 1%
```

Correct.

Very confident.

---

## Student B

```text
Dog      = 25%

Cat      = 60%

Horse    = 15%
```

Also correct.

But less confident.

---

## Student C

```text
Dog      = 97%

Cat      = 2%

Horse    = 1%
```

Wrong.

And extremely confident.

---

Question.

Should all three students receive the same score?

Of course not.

Student A deserves the highest score.

Student B is correct, but clearly uncertain.

Student C was not only wrong…

they were confidently wrong.

A good learning algorithm should recognize this difference.

---

# Chapter 2 — Accuracy Is Not Enough

Suppose we only measure accuracy.

Student A

```text
Prediction = Cat

Correct ✅
```

Accuracy = 100%

---

Student B

```text
Prediction = Cat

Correct ✅
```

Accuracy = 100%

---

According to accuracy,

both students performed equally well.

But did they?

Student A was almost certain.

Student B was barely confident.

Accuracy completely ignores confidence.

This is a serious problem.

Neural networks don't just need to know **whether** they are right.

They need to know **how right** they are.

---

# Chapter 3 — Softmax Already Did Half the Job

In the previous lecture, we learned that Softmax converts logits into probabilities.

Before Softmax

```text
Dog    = 4.2

Cat    = 7.8

Horse  = 2.5
```

After Softmax

```text
Dog    = 2.6%

Cat    = 97.0%

Horse  = 0.4%
```

This is great.

The model now tells us **how confident** it is.

But another question immediately appears.

> How do we tell the model whether these probabilities are good?

This is exactly where Cross Entropy enters.

---

# Chapter 4 — What Should a Good Loss Function Do?

Let's think like engineers.

If we were designing a loss function from scratch,

what properties would we want?

A good prediction should receive

```text
Small Loss
```

A bad prediction should receive

```text
Large Loss
```

A confidently correct prediction should receive

```text
Almost Zero Loss
```

A confidently wrong prediction should receive

```text
Very Large Loss
```

This sounds obvious.

But surprisingly,

Mean Squared Error doesn't behave this way for probability distributions.

Cross Entropy was specifically designed to measure the quality of probabilistic predictions.

---

# Chapter 5 — Understanding Cross Entropy Intuitively

Think of Cross Entropy as a teacher assigning penalties.

The model says,

> "I'm 98% sure the answer is Cat."

The teacher replies,

> "Excellent.
Very little penalty."

---

The model says,

> "I'm only 55% sure."

The teacher replies,

> "You're correct,
but I'd like you to become more confident."

Medium penalty.

---

The model says,

> "I'm 2% sure it's Cat."

The teacher replies,

> "That's not just wrong.

You were extremely confident about the wrong answer."

Huge penalty.

---

Notice something important.

Cross Entropy doesn't only care about correctness.

It also cares about confidence.

That is why it works so well for classification.

---

# Chapter 6 — Why Softmax and Cross Entropy Are Best Friends

Softmax produces probabilities.

Cross Entropy evaluates those probabilities.

They solve two completely different problems.

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

Cross Entropy

        │

        ▼

Loss
```

Without Softmax,

there are no probabilities.

Without probabilities,

Cross Entropy has nothing meaningful to evaluate.

This is why almost every modern classification model uses them together.

---

# Chapter 7 — Connecting This to GPT

Exactly the same process happens inside GPT.

Suppose the prompt is

```text
I like to eat an ______
```

The correct next token is

```text
apple
```

After Softmax,

GPT predicts

```text
apple      = 98%

banana     = 1%

car        = 0.5%

house      = 0.5%
```

Excellent prediction.

Very small loss.

---

Now suppose GPT predicts

```text
apple      = 2%

banana     = 95%

car        = 2%

house      = 1%
```

The prediction is completely wrong.

Even worse,

the model was extremely confident.

Cross Entropy assigns a very large penalty.

Gradient Descent then updates the weights so that next time,

the probability of **apple** increases,

while the probabilities of incorrect words decrease.

This is how GPT gradually learns language.

---

# Looking Ahead

So far we understand:

✅ Softmax converts logits into probabilities.

✅ Cross Entropy evaluates those probabilities.

One important question still remains.

How does Cross Entropy mathematically compute this penalty?

Why does it use the logarithm?

Why is the loss written as

$$
-\log(p)
$$

instead of something simpler?

---

# Chapter 8 — What Does Cross Entropy Actually Compare?

Until now, we've been saying:

> "Cross Entropy compares two probability distributions."

But what exactly are those two distributions?

Let's look at an example.

Suppose the correct answer is

```text
Cat
```

The model predicts

```text
Dog      = 2%

Cat      = 97%

Horse    = 1%
```

These probabilities came from Softmax.

This is called the **Predicted Distribution** because it represents what the model believes.

---

But what should the ideal prediction look like?

If the correct answer is **Cat**, then ideally the model should predict

```text
Dog      = 0%

Cat      = 100%

Horse    = 0%
```

This is called the **True Distribution**.

Notice something interesting.

The true distribution is also a probability distribution.

It sums to one.

Only one class has probability 1.

Every other class has probability 0.

Cross Entropy compares these two distributions.

```text
                 True Distribution

Dog      = 0

Cat      = 1

Horse    = 0

               vs

             Predicted Distribution

Dog      = 0.02

Cat      = 0.97

Horse    = 0.01
```

The closer these two distributions are,

the smaller the loss becomes.

---

# Chapter 9 — What Is One-Hot Encoding?

You might wonder,

> "Why does the true distribution contain only zeros and ones?"

This representation is called **One-Hot Encoding**.

Suppose we have three classes.

```text
Dog

Cat

Horse
```

If the correct answer is **Cat**,

the true distribution becomes

```text
Dog      = 0

Cat      = 1

Horse    = 0
```

Exactly one position contains a **1**.

Every other position contains **0**.

Hence the name

**One-Hot Encoding**.

---

## Why Is One-Hot Encoding Useful?

Imagine the model predicts

```text
Dog      = 0.02

Cat      = 0.97

Horse    = 0.01
```

The only probability we truly care about is

```text
Cat = 0.97
```

Because Cat is the correct answer.

The probabilities assigned to Dog and Horse don't directly contribute to the loss for this training example.

This greatly simplifies the Cross Entropy calculation.

---

# Chapter 10 — Why Does Only the Correct Class Matter?

Let's compare two predictions.

### Prediction A

```text
Dog      = 2%

Cat      = 97%

Horse    = 1%
```

---

### Prediction B

```text
Dog      = 1%

Cat      = 97%

Horse    = 2%
```

Notice something.

The probabilities assigned to Dog and Horse changed.

But the probability assigned to the correct class,

**Cat**,

remained exactly the same.

Should the loss change?

No.

For a single training example,

Cross Entropy only looks at the probability assigned to the **correct class**.

Everything else becomes irrelevant.

This is why you'll often hear people say:

> "Cross Entropy only cares about the probability of the correct class."

Technically, that's because the true distribution is one-hot encoded.

We'll prove this mathematically in the next lecture.

---

# Chapter 11 — The Intuition Behind the Formula

Suppose the correct answer is

```text
Cat
```

The model predicts

```text
Cat = 99%
```

Should the penalty be large?

Of course not.

The model did an excellent job.

Now suppose

```text
Cat = 60%
```

Still correct.

But the model isn't very confident.

A larger penalty seems reasonable.

Now suppose

```text
Cat = 2%
```

This is terrible.

The model was almost certain that another class was correct.

This deserves a huge penalty.

So we need a mathematical function that behaves like this:

| Probability of Correct Class | Desired Penalty |
|------------------------------|-----------------|
| 99% | Very Small |
| 90% | Small |
| 70% | Moderate |
| 50% | Large |
| 10% | Very Large |
| 1% | Huge |

Surprisingly,

the **negative logarithm** behaves exactly this way.

This is why Cross Entropy uses

$$
-\log(p)
$$

instead of a simple subtraction.

We'll derive this mathematically in the next lecture.

---

# Chapter 12 — The Simplified Cross Entropy Equation

Because the true distribution is one-hot encoded,

Cross Entropy simplifies to a remarkably elegant equation.

$$
Loss = -\log(P(\text{Correct Class}))
$$

Read this equation in plain English.

> "Take the probability that the model assigned to the correct answer.

> Compute its negative logarithm.

> That value becomes the loss."

Notice something beautiful.

Cross Entropy doesn't ask

> "Did you predict the correct class?"

Instead, it asks

> "How confident were you about the correct class?"

This subtle difference is what makes modern neural networks learn so effectively.

---

# Looking Ahead

We've now built the complete intuition behind Cross Entropy.

We understand:

✅ Why accuracy is not enough.

✅ Why Softmax converts logits into probabilities.

✅ Why Cross Entropy compares two probability distributions.

✅ What the true distribution represents.

✅ Why one-hot encoding is used.

✅ Why only the probability of the correct class matters.

The only mystery left is the mathematics.

In the next lecture, we'll derive the complete Cross Entropy formula step by step.

We'll answer questions like:

- Why does the logarithm appear?
- Why is there a negative sign?
- Why is the complete equation written as

$$
-\sum y \log(\hat{y})
$$

instead of simply

$$
-\log(p)
$$

By the end of that lecture, you'll not only know the formula—you'll understand why it had to be this way.