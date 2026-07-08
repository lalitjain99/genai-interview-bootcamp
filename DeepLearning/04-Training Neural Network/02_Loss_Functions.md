# Loss Functions — Turning Mistakes into Numbers

## Module 04 — Training Neural Networks

## Learning Objectives

By the end of this lecture, you should be able to answer:

- What is a loss function?
- Why can't a neural network learn without a loss function?
- What makes a good loss function?
- Why are there different loss functions?
- What is the difference between Loss and Accuracy?
- Why do regression and classification require different loss functions?
- How does GPT use a loss function?

---

# Chapter 1 — A Teacher Needs More Than "Right" or "Wrong"

Imagine you're teaching two students.

Both answered the same question.

The correct answer is:

```text
Paris
```

---

### Student A

```text
Answer: Paris
Confidence: 99%
```

Excellent.

---

### Student B

```text
Answer: Paris
Confidence: 51%
```

Also correct.

---

Now another student.

### Student C

```text
Answer: London
Confidence: 98%
```

Completely wrong.

---

As a teacher, you immediately know:

- Student A performed best.
- Student B is correct but uncertain.
- Student C is confidently wrong.

Simply saying

```text
Right

or

Wrong
```

isn't enough.

You need a way to measure **how good** or **how bad** every prediction was.

That measurement is called the **Loss**.

---

# Chapter 2 — What Is a Loss Function?

A loss function is a mathematical function that compares

```text
Prediction
```

with

```text
Ground Truth
```

and produces a single number.

```text
Prediction

↓

Loss Function

↓

One Number
```

That number answers one question:

> **"How bad was this prediction?"**

---

Small mistake

↓

Small loss

---

Large mistake

↓

Large loss

---

Perfect prediction

↓

Zero (or nearly zero) loss

---

Think of it as a teacher assigning marks.

Instead of saying,

> "Wrong."

the teacher says,

> "You lost 8 marks."

That number gives much richer feedback.

---

# Chapter 3 — Why Convert Mistakes into Numbers?

Imagine trying to improve without knowing how much you improved.

Suppose yesterday your prediction error was

```text
20
```

Today it became

```text
8
```

Tomorrow it became

```text
2
```

You can immediately see progress.

A neural network learns exactly the same way.

Its goal is simple.

```text
Reduce the Loss

↓

Improve Predictions

↓

Reduce the Loss Again

↓

Repeat
```

Training is nothing more than repeatedly reducing this number.

---

# Chapter 4 — What Makes a Good Loss Function?

Not every way of measuring mistakes is useful.

A good loss function should satisfy several properties.

### 1. Perfect predictions should have very small loss.

If the prediction is exactly correct,

the model should receive almost no penalty.

---

### 2. Bigger mistakes should receive larger penalties.

A prediction that is far from the correct answer should have a much larger loss than a prediction that is only slightly wrong.

---

### 3. The loss should change smoothly.

Imagine climbing down a mountain.

If the slope is smooth,

it's easy to know which direction to walk.

If the mountain has sudden cliffs,

finding the correct direction becomes difficult.

Gradient Descent needs a smooth landscape to find better weights.

---

### 4. The loss should guide learning.

A loss function is not just a score.

It is also a guide.

It tells the optimization algorithm

which direction will reduce future mistakes.

---

# Chapter 5 — Why Do We Have Different Loss Functions?

Suppose you're predicting

House Price.

Should being off by ₹10,000 be treated the same as predicting the wrong animal?

Of course not.

Different problems require different ways of measuring mistakes.

---

## Regression

Output

```text
₹52,35,000
```

Mistake

```text
Predicted Price

vs

Actual Price
```

Common loss functions

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

---

## Classification

Output

```text
Dog

Cat

Horse
```

Mistake

```text
Predicted Probability

vs

Correct Class
```

Common loss function

- Cross Entropy

---

The loss function depends on **what the model is trying to predict.**

---

# Chapter 6 — Loss vs Accuracy

One of the most common interview questions is:

> "If accuracy tells us how good the model is, why don't we use accuracy as the loss function?"

Let's compare them.

| Accuracy | Loss |
|----------|------|
| Measures correctness | Measures the magnitude of mistakes |
| Used for evaluation | Used for learning |
| Doesn't measure confidence | Measures confidence |
| Not differentiable | Usually differentiable |

Accuracy answers:

> "Was the prediction correct?"

Loss answers:

> "How wrong was the prediction?"

Neural networks need the second answer during training.

---

# Chapter 7 — Connecting Everything

The complete training pipeline now looks like this.

```text
Input

↓

Neural Network

↓

Prediction

↓

Loss Function

↓

Loss

↓

Gradient Descent

↓

Updated Weights

↓

Better Prediction

↓

Repeat
```

Notice that the loss function sits exactly between

the prediction

and

the learning algorithm.

It acts as the bridge between them.

Without a loss function,

Gradient Descent would have no idea what to optimize.

---

# Chapter 8 — GPT Connection

GPT follows exactly the same learning pipeline.

Suppose the input is

```text
The capital of France is
```

The correct next token is

```text
Paris
```

GPT predicts

```text
London
```

The loss function compares

```text
Predicted Token

↓

Correct Token

↓

Loss
```

Cross Entropy produces a large loss because the prediction is both wrong and highly confident.

That loss is then used to update billions of parameters inside the Transformer.

The process repeats billions of times until the model gradually learns language.

---

# Looking Ahead

Now we understand what a loss function is.

The next question naturally becomes:

> "How should we measure numerical mistakes?"

Should we simply subtract the values?

Should we square the error?

Should we ignore the sign?

These questions lead us to one of the simplest and most important loss functions in machine learning:

**Mean Squared Error (MSE).**