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

# 🧠 Engineer Discussion — Thinking Like an ML Engineer

The following thought experiments are designed to test your understanding of **Mean Squared Error (MSE)**.

Rather than memorizing the formula, try to reason about how MSE influences the learning process.

---

# Question 1 — Why Do We Square the Error?

Suppose we predict house prices.

```text
House A
Actual Price      = ₹100
Predicted Price   = ₹120

Error = +20
```

```text
House B
Actual Price      = ₹100
Predicted Price   = ₹80

Error = -20
```

## Question

Why don't we simply average the raw errors?

Why do we square them?

---

## Answer

If we average the raw errors,

```text
(+20) + (-20) = 0
```

The model appears to have made **zero average error**, even though both predictions are incorrect.

This is called **error cancellation**.

Squaring the errors solves this problem because

```text
(+20)² = 400

(-20)² = 400
```

Both mistakes now contribute positively to the loss.

Squaring also gives another important property:

```text
Small mistakes receive small penalties.

Large mistakes receive much larger penalties.
```

Finally, squaring produces a smooth, differentiable loss surface that allows Gradient Descent to learn efficiently.

---

# Question 2 — Why Can't Loss Be Negative?

Suppose

```text
Actual = 10

Prediction = 8

Error = -2
```

## Question

Why shouldn't the loss be allowed to become negative?

---

## Answer

Loss measures **how bad** a prediction is.

Negative badness has no practical meaning.

The sign of the error already tells us whether the prediction is above or below the target.

The loss only measures **the magnitude of the mistake**.

Allowing positive and negative losses would again create the problem of error cancellation.

---

# Question 3 — Which Model Does MSE Prefer?

Consider two models.

Model A

```text
Errors

1
2
1
2
```

Model B

```text
Errors

0
0
0
6
```

## Question

Without calculating the MSE,

which model will MSE prefer?

Why?

---

## Answer

MSE prefers **Model A**.

Although Model B predicts three values perfectly,

its single large mistake dominates the squared loss.

This reveals an important property of MSE:

```text
MSE rewards consistency.

It strongly penalizes large outliers.
```

This is exactly what we want in safety-critical applications like autonomous driving or robotics, where one catastrophic prediction is often worse than several small mistakes.

---

# Question 4 — Why Is MSE Preferred for Self-Driving Cars?

Suppose you are designing the steering controller for an autonomous car.

Would you choose

- Mean Absolute Error (MAE)

or

- Mean Squared Error (MSE)?

---

## Answer

MSE is usually the better choice.

A steering prediction that is off by

```text
1°
```

is acceptable.

A prediction that is off by

```text
40°
```

could cause a serious accident.

Because MSE squares the error,

large steering mistakes receive much larger penalties.

The model therefore learns to avoid catastrophic errors.

---

# Question 5 — Does a Lower MSE Mean the Model Is Twice as Good?

Suppose

```text
Initial MSE = 250

Final MSE = 125
```

## Question

Can we conclude that the model is now twice as good?

---

## Answer

No.

MSE is not a linear measure of model quality.

Since the error is squared,

reducing the MSE by half does not imply the predictions became twice as accurate.

Loss values should be interpreted as optimization objectives, not percentages of model quality.

---

# Question 6 — What Happens If We Scale the Dataset?

Suppose every target value becomes

```text
100 × larger
```

## Question

What happens to the MSE?

---

## Answer

Every prediction error also becomes

```text
100 × larger
```

Since MSE squares every error,

```text
(100 × Error)²

=

10,000 × Error²
```

Therefore,

```text
MSE becomes 10,000 times larger.
```

---

# Question 7 — What If We Didn't Square the Error?

Suppose we define

```text
Loss = Average(Error)
```

instead of

```text
Loss = Average(Error²)
```

## Question

Would Gradient Descent still learn correctly?

---

## Answer

No.

There are two major problems.

### Error Cancellation

Positive and negative errors cancel each other.

```text
+20

and

-20

↓

Average = 0
```

The model appears perfect even though both predictions are wrong.

### Constant Gradient

The derivative of a raw error is constant.

That means the optimizer receives nearly the same update regardless of whether the prediction is slightly wrong or extremely wrong.

The model loses the ability to distinguish between small and large mistakes.

---

# Question 8 — Why Does MSE Care So Much About Outliers?

Suppose

```text
999 samples
```

have very small errors,

while

```text
1 sample
```

has a very large error.

## Question

How will MSE behave?

---

## Answer

The large error will dominate the loss.

This is intentional.

In applications like

- autonomous driving
- robotics
- aircraft control
- medical diagnosis

one catastrophic prediction is much more dangerous than many small mistakes.

MSE therefore forces the model to pay special attention to these large errors.

---

# Question 9 — Why Do We Take the Mean Instead of the Sum?

Suppose two datasets contain

```text
Dataset A

100 samples
```

and

```text
Dataset B

1,000,000 samples
```

## Question

Why don't we simply sum the squared errors?

---

## Answer

The total squared error naturally grows as the dataset becomes larger.

A larger dataset would always produce a larger loss,

even if both models have exactly the same prediction quality.

Taking the average makes the loss independent of the dataset size.

This allows fair comparison across different datasets and different mini-batches during training.

---

# Question 10 — Why Does MSE Naturally Guide Learning?

Suppose the prediction gradually improves.

```text
Prediction

90

↓

95

↓

99

↓

99.9

↓

100
```

## Question

How should the gradients behave during this journey?

Why is this desirable?

---

## Answer

As the prediction becomes more accurate,

the error becomes smaller.

A smaller error produces a smaller gradient.

Therefore,

```text
Large Error

↓

Large Gradient

↓

Large Weight Update
```

As training progresses,

```text
Small Error

↓

Small Gradient

↓

Small Weight Update
```

This is exactly the behavior we want.

When the model is far from the correct solution,

it learns quickly by taking large optimization steps.

As it approaches the optimum,

the updates naturally become smaller,

allowing the model to converge smoothly without overshooting.

This is what we mean when we say:

> **A good loss function should guide the learning process.**

The loss itself does not tell the model where to move.

Its **gradient** tells the optimizer both

- the direction to move, and
- how large the update should be.

---

# Key Engineering Takeaways

- MSE prevents error cancellation.
- Large mistakes receive disproportionately larger penalties.
- MSE produces a smooth, differentiable loss surface.
- MSE rewards consistent predictions.
- MSE naturally reduces the update size as the model approaches the optimum.
- MSE guides learning through its gradients, not through its numerical value.
- MSE is particularly suitable for applications where large mistakes are unacceptable.
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