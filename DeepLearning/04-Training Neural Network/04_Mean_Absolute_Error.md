# Mean Absolute Error (MAE) — When Squaring the Error Becomes a Problem

## Module 04 — Training Neural Networks

## Learning Objectives

By the end of this lecture, you should be able to answer:

- Why isn't Mean Squared Error always the best choice?
- What are outliers?
- Why does MSE become sensitive to outliers?
- What is Mean Absolute Error (MAE)?
- How is MAE calculated?
- When should we prefer MAE over MSE?
- What are the advantages and limitations of MAE?

---

# Chapter 1 — The Teacher Notices Something Strange

Yesterday we learned that squaring the error has two major advantages.

✅ Negative errors become positive.

✅ Large mistakes receive larger penalties.

Everything looked perfect.

But imagine you're a teacher checking exam marks.

The correct marks are:

```text
80
82
78
81
79
```

Most students are very close to the correct answer.

Now imagine one student writes

```text
10
```

This student made an enormous mistake.

Now ask yourself.

> Should one terrible prediction dominate the evaluation of the entire class?

This is where MSE starts showing its weakness.

---

# Chapter 2 — Meet the Outlier

A value that is very different from the rest of the data is called an **Outlier**.

For example,

```text
10

12

11

13

95
```

Here,

```text
95
```

is an outlier.

It is much farther away from the remaining values.

Outliers appear everywhere in the real world.

Examples:

- Sensor malfunction
- Human data entry mistakes
- Fraudulent transactions
- Network failures
- Rare but extreme events

A good loss function should behave reasonably even when such unusual values appear.

---

# Chapter 3 — How MSE Reacts to an Outlier

Suppose the correct house price is

```text
₹1 Crore
```

Model A predicts

```text
₹99 Lakhs
```

Error

```text
₹1 Lakh
```

Squared Error

```text
1² = 1
```

---

Now another prediction

```text
₹40 Lakhs
```

Error

```text
₹60 Lakhs
```

Squared Error

```text
60² = 3600
```

Notice what happened.

The error became

```text
60 times larger
```

But the penalty became

```text
3600 times larger
```

Squaring magnifies large errors dramatically.

Sometimes this is exactly what we want.

Sometimes it is not.

---

# Chapter 4 — A Different Philosophy

Imagine another teacher.

Instead of saying,

> "Huge mistakes deserve enormous punishment,"

this teacher says,

> "Every mistake should be treated proportionally."

Whether the mistake is

```text
2
```

or

```text
20
```

the penalty should grow linearly.

Not quadratically.

This idea leads us to a different loss function.

---

# Chapter 5 — Mean Absolute Error

Instead of squaring the error,

we simply ignore the sign.

Mathematically,

$$
MAE=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y_i}|
$$

Instead of

```text
(Error)²
```

we calculate

```text
|Error|
```

This means

```text
-20

↓

20
```

and

```text
20

↓

20
```

Negative errors no longer cancel positive ones,

but large errors are **not exaggerated**.

---

# Chapter 6 — Worked Example

Suppose the actual prices are

```text
100

200

300
```

Predictions

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

### Step 2 — Take Absolute Values

```text
10

20

20
```

---

### Step 3 — Average

```text
(10 + 20 + 20) / 3

= 16.67
```

The Mean Absolute Error is

```text
16.67
```

Simple.

Easy to interpret.

---

# Chapter 7 — MSE vs MAE

Now let's compare them.

Suppose the error is

```text
2
```

MSE penalty

```text
2² = 4
```

MAE penalty

```text
2
```

---

Suppose the error becomes

```text
10
```

MSE penalty

```text
10² = 100
```

MAE penalty

```text
10
```

Notice the difference.

MSE grows much faster.

MAE grows steadily.

---

# Chapter 8 — Which One Is Better?

This is the wrong question.

Neither is universally better.

It depends on the problem.

### Prefer MSE when

- Large mistakes are very expensive.
- You want the model to strongly avoid large errors.
- Outliers are meaningful and should influence training.

Examples:

- Medical diagnosis
- Autonomous driving
- Safety-critical systems

---

### Prefer MAE when

- Outliers are common.
- You want a robust model.
- Every error should contribute proportionally.

Examples:

- Financial forecasting
- Sensor data with occasional failures
- Real-world noisy datasets

---

# Chapter 9 — Why Doesn't Everyone Use MAE?

If MAE is robust to outliers,

why isn't it always preferred?

Because MAE has its own limitation.

Remember when we discussed activation functions?

Gradient Descent works best when the optimization landscape is smooth.

MSE produces a smooth, continuously changing loss surface.

MAE contains sharp corners (because of the absolute value).

These sharp points make optimization more challenging.

We'll understand this much more clearly when we study Gradient Descent.

For now, remember:

> MAE is more robust.

> MSE is usually easier to optimize.

---

# Chapter 10 — GPT Connection

GPT predicts probabilities, not continuous numbers.

Its goal is not to estimate a house price or a temperature.

Instead, it predicts the probability of the next token.

Therefore,

neither MSE nor MAE is commonly used for training GPT.

GPT instead uses **Cross Entropy Loss**, which is specifically designed for comparing probability distributions.

---

# Mental Model

```text
Error

↓

Take Absolute Value

↓

Average All Errors

↓

Mean Absolute Error

↓

Loss

↓

Gradient Descent

↓

Better Model
```

---

# MSE vs MAE Summary

| Property | MSE | MAE |
|----------|-----|-----|
| Formula | Error² | Absolute Error |
| Penalizes Large Errors | Very Strongly | Linearly |
| Sensitive to Outliers | Yes | No |
| Smooth Optimization | Yes | Less Smooth |
| Common Use | Regression | Regression with Outliers |

---

# Looking Ahead

We have now studied two important loss functions for regression.

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

But classification is a completely different problem.

When the output is

```text
Dog

Cat

Horse
```

there is no meaningful numerical distance between classes.

Instead of measuring numerical error,

we need to compare **probability distributions**.

This brings us to one of the most important loss functions in Deep Learning:

**Cross Entropy Loss.**