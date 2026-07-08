# Mean Squared Error (MSE) — Why Do We Square the Error?

## Module 04 — Training Neural Networks

## Learning Objectives

By the end of this lecture, you should be able to answer:

- Why can't we simply subtract the prediction from the actual value?
- Why do we need a mathematical way to measure error?
- What problems arise with positive and negative errors?
- Why do we square the error instead of taking its absolute value?
- What is Mean Squared Error (MSE)?
- Where is MSE used in Machine Learning?
- Why isn't MSE commonly used for classification?

---

# Chapter 1 — The Teacher's Report Card

Imagine you're a mathematics teacher.

Three students attempt the same question.

The correct answer is:

```text
100
```

The students answer:

```text
Student A = 98

Student B = 105

Student C = 150
```

Now you want to measure **how wrong** each student was.

Your first instinct might be simple.

```text
Error = Actual - Prediction
```

Let's calculate.

```text
Student A

100 - 98 = +2
```

```text
Student B

100 - 105 = -5
```

```text
Student C

100 - 150 = -50
```

Looks reasonable.

But something strange happens next.

---

# Chapter 2 — The Cancellation Problem

Suppose two predictions are made.

```text
Prediction 1

Error = +20
```

```text
Prediction 2

Error = -20
```

Now compute the average error.

```text
(+20) + (-20)

= 0
```

According to mathematics,

our model made **zero average error**.

But is that true?

Absolutely not.

The model made two large mistakes.

They simply cancelled each other.

Positive and negative errors hide the real performance of the model.

This is called the **Error Cancellation Problem**.

A good loss function should never allow large mistakes to disappear just because they have opposite signs.

---

# Chapter 3 — First Attempt: Ignore the Sign

One obvious solution is:

Take the absolute value.

```text
|Actual - Prediction|
```

Now our previous example becomes:

```text
|+20| = 20

|-20| = 20
```

Average Error

```text
20
```

Much better.

No cancellation.

So...

Have we solved the problem?

Not completely.

There is still another issue waiting for us.

---

# Chapter 4 — Should Large Mistakes Hurt More?

Imagine two students.

Student A misses the correct answer by

```text
2 marks
```

Student B misses by

```text
40 marks
```

Should Student B receive only twenty times the penalty?

Or should we punish such a huge mistake much more heavily?

In many real-world applications, large mistakes are far more dangerous than small ones.

Imagine predicting:

```text
Actual House Price

₹1 Crore
```

Prediction A

```text
₹99 Lakhs
```

Error = ₹1 Lakh

---

Prediction B

```text
₹40 Lakhs
```

Error = ₹60 Lakhs

Clearly,

the second prediction is much worse.

A good loss function should emphasize these large mistakes.

---

# Chapter 5 — Squaring the Error

Instead of taking the absolute value,

what if we square the error?

Let's see.

Small mistake

```text
Error = 2

Squared Error

2² = 4
```

Large mistake

```text
Error = 10

Squared Error

10² = 100
```

Notice something interesting.

The larger mistake grows much faster.

Squaring achieves two important goals simultaneously:

✅ Negative errors become positive.

✅ Large mistakes receive a much larger penalty.

This makes squaring an excellent choice for many regression problems.

---

# Chapter 6 — From Squared Error to Mean Squared Error

So far, we've measured the error for one prediction.

But during training,

a neural network makes thousands—or even millions—of predictions.

We therefore calculate the squared error for every prediction and then compute the average.

This gives us the **Mean Squared Error**.

Mathematically,

$$
MSE=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
$$

where:

- **n** = Number of samples
- **y** = Actual value
- **ŷ** = Predicted value

---

# Chapter 7 — Understanding the Formula

Let's decode the formula step by step.

### Step 1

Find the error.

```text
Actual - Prediction
```

---

### Step 2

Square the error.

```text
(Error)²
```

---

### Step 3

Repeat for every sample.

---

### Step 4

Add all squared errors.

---

### Step 5

Divide by the number of samples.

The result is one number.

That number tells us how well the model performed overall.

---

# Chapter 8 — Worked Example

Suppose the actual prices are

```text
100

200

300
```

The model predicts

```text
90

220

280
```

### Step 1 — Calculate Errors

```text
10

-20

20
```

---

### Step 2 — Square Them

```text
100

400

400
```

---

### Step 3 — Average

```text
(100 + 400 + 400) / 3

= 300
```

The Mean Squared Error is

```text
300
```

Notice that the negative error did not cancel the positive ones.

---

# Chapter 9 — Why MSE Works Well for Regression

Regression predicts continuous values.

Examples:

- House prices
- Temperature
- Stock prices
- Rainfall
- Sales Forecasting

These problems care about **how far** the prediction is from the actual value.

MSE measures exactly that.

This is why MSE became one of the most popular regression loss functions.

---

# Chapter 10 — Why MSE Is Not Ideal for Classification

Suppose we are classifying animals.

Correct answer

```text
Cat
```

Prediction

```text
Dog
```

Can we compute

```text
Cat - Dog
```

Of course not.

Categories don't have meaningful numerical distances.

Classification problems need to compare **probabilities**, not numeric values.

That is why classification usually uses **Cross Entropy** instead of MSE.

---

# Chapter 11 — GPT Connection

GPT does not predict house prices or temperatures.

It predicts probabilities for the next token.

For example,

```text
apple = 92%

banana = 5%

car = 2%

house = 1%
```

The problem is no longer

> "How far is the prediction from the actual number?"

Instead, it becomes

> "How close is the predicted probability distribution to the correct one?"

This is why GPT uses **Cross Entropy Loss** rather than Mean Squared Error.

---

# Mental Model

```text
Prediction

↓

Calculate Error

↓

Square the Error

↓

Average All Errors

↓

Mean Squared Error

↓

Loss

↓

Gradient Descent

↓

Better Model
```

---

# Looking Ahead

Mean Squared Error solved two important problems:

- It prevents positive and negative errors from cancelling each other.
- It penalizes large mistakes more heavily than small ones.

But another question naturally arises.

> **Why square the error? Why not simply use the absolute value?**

Sometimes, squaring large errors too aggressively can become a disadvantage.

That leads us to another important loss function:

**Mean Absolute Error (MAE).**