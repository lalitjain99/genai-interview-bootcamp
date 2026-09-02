# How Does a Neural Network Learn?

## Module 04 — Training Neural Networks

## Learning Objectives

By the end of this lecture, you should be able to answer:

- Why isn't making predictions enough?
- How does a neural network know it made a mistake?
- What is feedback in machine learning?
- Why do we need a loss function?
- What happens after a prediction is made?
- Why is learning an iterative process?
- How does this connect to GPT?

---

# Chapter 1 — A Student Without a Teacher

Imagine a classroom.

A student walks into the exam hall and answers every question.

When the exam ends, the student submits the paper.

Now imagine something strange happens.

The teacher never checks the paper.

No marks are given.

No corrections.

No feedback.

The student simply receives another exam the next day.

What do you think will happen?

Will the student improve?

Of course not.

The student has no idea whether the previous answers were right or wrong.

Without feedback, there is no learning.

---

# Chapter 2 — A Neural Network Faces the Same Problem

A neural network is surprisingly similar.

Suppose we build a model to recognize handwritten digits.

It receives an image.

```text
      Image
        │
        ▼
Neural Network
        │
        ▼
Prediction = 3
```

But the correct answer was

```text
5
```

Now ask yourself a simple question.

> How does the neural network know it was wrong?

The network doesn't have eyes.

It doesn't know the correct answer.

It only knows the prediction it made.

Someone has to compare the prediction with reality.

Someone has to say,

> "You predicted 3."

> "The correct answer was 5."

Only then can learning begin.

---

# Chapter 3 — The Missing Teacher

In machine learning, we don't have a human teacher checking millions of predictions.

Instead, we use mathematics.

After every prediction, we calculate a number.

This number answers one simple question:

> "How bad was this prediction?"

If the prediction was excellent,

the number should be very small.

If the prediction was terrible,

the number should be very large.

This number is called the **Loss**.

---

# Chapter 4 — What Is a Loss Function?

A loss function is simply a mathematical teacher.

It compares

```text
Prediction
```

with

```text
Ground Truth
```

and assigns a penalty.

Small mistake

↓

Small penalty

Large mistake

↓

Large penalty

Perfect prediction

↓

Almost zero penalty

---

Think of the loss function as a strict teacher.

A careless mistake might lose one mark.

A completely wrong answer might lose ten marks.

The worse the prediction,

the larger the penalty.

---

# Chapter 5 — Why Accuracy Isn't Enough

You might wonder,

> "Why don't we simply calculate accuracy?"

Let's imagine two students.

Student A predicts

```text
Cat = 99%

Dog = 1%
```

Correct.

---

Student B predicts

```text
Cat = 51%

Dog = 49%
```

Also correct.

According to accuracy,

both students score

```text
100%
```

But should they receive the same feedback?

No.

Student A was highly confident.

Student B was almost guessing.

A learning algorithm needs richer feedback than just

"right"

or

"wrong."

That is why we use loss functions instead of accuracy during training.

---

# Chapter 6 — Learning Is a Cycle

Training is not a single prediction.

It is a continuous loop.

```text
Input

↓

Prediction

↓

Compare with Ground Truth

↓

Calculate Loss

↓

Learn from Mistake

↓

Update Weights

↓

Make Better Prediction

↓

Repeat
```

Notice something important.

The network is never told

> "Increase weight number 17."

It is only told

> "Your prediction was bad."

Everything else must be discovered automatically.

That is the beauty of machine learning.

---

# Chapter 7 — A Baby Learning to Walk

Think about a baby learning to walk.

The baby stands.

Falls.

Adjusts balance.

Tries again.

Falls less.

Adjusts again.

Eventually...

The baby walks.

No one tells the baby

> "Move your left ankle by exactly 2.3 degrees."

The baby only receives feedback.

That feedback gradually changes future behavior.

Neural networks learn in exactly the same way.

Prediction.

Feedback.

Adjustment.

Repeat.

---

# Chapter 8 — Connecting This to GPT

Exactly the same process happens when GPT learns language.

Suppose the training sentence is

```text
The sky is blue.
```

GPT predicts the next word.

Instead of

```text
blue
```

it predicts

```text
green
```

The prediction is wrong.

A loss function measures **how wrong** it was.

That loss becomes the learning signal.

Later,

Gradient Descent and Backpropagation use this signal to update millions—or even billions—of weights.

After seeing billions of examples,

GPT gradually becomes better at predicting the next word.

---

# Chapter 9 — Where Does GPT Get the Ground Truth?

By now, you might have an important question.

> "For traditional Machine Learning, I understand where the ground truth comes from. We create a dataset with input features (**X**) and correct outputs (**Y**). But GPT is trained on billions of web pages, books, and articles. Nobody manually labels all that data. So where does the ground truth come from?"

This is one of the biggest differences between traditional Machine Learning and Large Language Models.

---

## Traditional Machine Learning

Suppose we want to predict house prices.

Our dataset already contains both the input and the correct answer.

| Size (sq.ft.) | Bedrooms | Price |
|---------------|----------|-------|
| 1000 | 2 | ₹50 Lakhs |
| 1200 | 3 | ₹65 Lakhs |
| 1500 | 3 | ₹80 Lakhs |

Here,

```text
Input (X)

↓

Model

↓

Predicted Price

↓

Compare with Actual Price (Y)

↓

Loss
```

The ground truth is explicitly provided in the dataset.

---

## GPT Is Different

Now imagine GPT is reading the following sentence:

```text
The capital of France is Paris.
```

Where is the label?

Where is the correct output?

At first glance, it looks like there isn't one.

So how can GPT learn?

The answer is surprisingly clever.

GPT creates its own training examples from the text itself.

---

## Turning One Sentence into Many Training Examples

Instead of treating the entire sentence as one example, GPT breaks it into many smaller prediction tasks.

### Example 1

Input

```text
The
```

Ground Truth

```text
capital
```

---

### Example 2

Input

```text
The capital
```

Ground Truth

```text
of
```

---

### Example 3

Input

```text
The capital of
```

Ground Truth

```text
France
```

---

### Example 4

Input

```text
The capital of France
```

Ground Truth

```text
is
```

---

### Example 5

Input

```text
The capital of France is
```

Ground Truth

```text
Paris
```

---

### Example 6

Input

```text
The capital of France is Paris
```

Ground Truth

```text
.
```

From just one sentence, GPT automatically creates multiple labeled training examples.

No human had to manually provide the answers.

The sentence itself contains the correct next word.

---

## Why Is This Called Self-Supervised Learning?

Unlike traditional supervised learning, nobody manually creates input-output pairs.

Instead, the data supervises itself.

```text
Original Sentence

↓

Hide the Next Word

↓

Ask the Model to Predict It

↓

Compare with the Actual Next Word

↓

Compute Loss
```

The "next word" already exists in the original text.

That becomes the ground truth.

This learning paradigm is called **Self-Supervised Learning** because the supervision comes directly from the data itself.

---

## How Does the Loss Function Work Here?

Suppose GPT receives the input:

```text
The capital of France is
```

The correct next token is:

```text
Paris
```

But GPT predicts:

```text
London
```

After applying Softmax, the model assigns the following probabilities:

```text
Paris   = 4%

London  = 80%

Berlin  = 10%

Rome    = 6%
```

The ground truth is still **Paris**.

Cross Entropy compares the predicted probability for the correct token (**Paris**) with the actual answer and computes the loss.

A large loss tells the model,

> "You were highly confident, but your prediction was wrong."

This loss signal is later used by Gradient Descent and Backpropagation to update the model's weights.

---

## Traditional ML vs GPT Training

```text
Traditional Machine Learning

Dataset

↓

(X, Y) provided by humans

↓

Prediction

↓

Loss
```

```text
GPT Training

Raw Text

↓

Automatically create (Input, Next Token) pairs

↓

Prediction

↓

Loss
```

Both approaches use a loss function.

The only difference is **where the ground truth comes from**.

- In traditional Machine Learning, humans provide the labels.
- In GPT, the text itself provides the labels.

---

## Interview Insight

A common interview question is:

> **If GPT is trained on unlabeled text, why is its training called self-supervised learning instead of unsupervised learning?**

The answer is:

Although humans do not manually label the data, the training process still has a correct answer for every prediction.

The next token in the sentence acts as the ground truth.

Since the labels are automatically generated from the data itself, the process is called **Self-Supervised Learning**, not unsupervised learning.

---

# Looking Ahead

Today we answered an important question.

> How does a neural network know it made a mistake?

The answer is:

A **loss function** compares the prediction with the correct answer and produces a numerical penalty.

But another question immediately appears.

> How should we calculate this penalty?

Should we simply subtract the values?

Should we square the error?

Should we use absolute values?

Or is there a better approach?

That is exactly what we'll begin exploring in the next lecture.

We'll start with one of the simplest and most widely used loss functions:

**Mean Squared Error (MSE).**