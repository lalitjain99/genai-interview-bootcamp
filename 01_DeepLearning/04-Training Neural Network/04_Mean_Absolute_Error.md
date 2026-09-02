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

## 💡 Engineer's Insight

Two models can have exactly the same MAE while making completely different kinds of mistakes.

For example:

Model A

```text
Errors = [5, 5, 5, 5, 5]

MAE = 5
```

Model B

```text
Errors = [0, 0, 0, 0, 25]

MAE = 5
```

Although both models have the same MAE, their behavior is very different.

- Model A makes small, consistent errors.
- Model B is perfect most of the time but occasionally makes a very large mistake.

This demonstrates that MAE summarizes the average magnitude of errors but does **not** describe how those errors are distributed.

When evaluating a machine learning model, it is often useful to examine both the loss value and the distribution of errors.

## 💡 Staff Engineer Insight

A Staff Engineer does not choose a loss function based only on its mathematical properties.

Instead, they ask:

- What business metric are we optimizing?
- Are large mistakes expensive?
- Are outliers meaningful or simply noise?
- What is the real cost of making a prediction error?

The same dataset may require different loss functions depending on the business objective.

For example:

- If the goal is consistent average performance across thousands of predictions, MAE may be preferred because it is robust to outliers.

- If rare but large mistakes have significant financial or safety consequences, MSE may be preferred because it penalizes large errors much more heavily.

A good engineering decision aligns the optimization objective with the business objective.


---

# 🧠 Engineer Discussion & Intuitive Q&A

## Q1 — Should Positive and Negative Errors Be Treated Equally?

Suppose your model predicts the price of a house.

```text
Actual Price = ₹100

Prediction A = ₹90

Prediction B = ₹110
```

Should Mean Absolute Error (MAE) assign the same loss to both predictions?

### Answer

Yes.

MAE computes the **absolute difference** between the prediction and the target.

```text
|100 - 90| = 10

|100 - 110| = 10
```

Since both predictions are equally far from the correct answer, MAE assigns them the same penalty.

MAE only considers **how large the mistake is**, not whether the prediction is above or below the target.

---

## Q2 — Why Do We Use Absolute Error Instead of Raw Error?

Suppose the correct value is

```text
100
```

Two predictions are

```text
Prediction 1 = 90

Prediction 2 = 110
```

Using raw error,

```text
Error 1 = -10

Error 2 = +10
```

### Answer

If we average the raw errors,

```text
(-10 + 10) / 2 = 0
```

The model would incorrectly conclude that it made **zero average error**, even though both predictions were wrong.

This is known as **error cancellation**.

Taking the absolute value ensures that:

- Positive and negative errors cannot cancel each other.
- The loss is always non-negative.
- Every mistake contributes to learning.

---

## Q3 — How Does MAE Handle Outliers?

Suppose a dataset contains

```text
99 predictions

Error = 1
```

and

```text
1 prediction

Error = 100
```

Which loss function is more affected by this outlier?

### Answer

MAE applies a **linear penalty**.

```text
Error = 100

↓

Penalty = 100
```

MSE applies a **quadratic penalty**.

```text
Error = 100

↓

Penalty = 10,000
```

Therefore, the single outlier dominates the overall loss much more in MSE than in MAE.

This makes MAE much more robust to noisy or unusual data.

---

## Q4 — Is MAE Always Better for Large Errors?

Suppose you are building:

- A house price prediction model
- A movie recommendation system

Should you automatically choose MAE because it is robust to outliers?

### Answer

No.

The correct choice depends on the **business objective**, not the application domain.

Ask questions such as:

- Are large prediction errors acceptable?
- Should one large mistake dominate learning?
- Are outliers genuine or simply noise?

If large errors have severe consequences, MSE may be preferable.

If outliers are mostly noise, MAE is often a better choice.

Engineering decisions should be driven by business requirements rather than mathematical preference.

---

## Q5 — Can Two Models Have the Same MAE but Behave Very Differently?

Consider two models.

Model A

```text
Errors = [5, 5, 5, 5, 5]

MAE = 5
```

Model B

```text
Errors = [0, 0, 0, 0, 25]

MAE = 5
```

### Answer

Yes.

Although both models have the same MAE, their behavior is very different.

- Model A makes small, consistent mistakes.
- Model B is perfect most of the time but occasionally makes a very large mistake.

MAE summarizes only the **average magnitude** of errors.

It does not describe how those errors are distributed.

---

## Q6 — What Problem Does MAE Have During Optimization?

The MAE loss is

```text
|y - ŷ|
```

Imagine the graph of

```text
|x|
```

### Answer

The graph contains a **sharp corner** at

```text
x = 0
```

At this point, the derivative is **undefined**.

Since Gradient Descent relies on derivatives to determine the direction of learning, this sharp corner makes optimization more difficult than with smooth loss functions like MSE.

Although MAE can still be optimized in practice using subgradients, MSE generally provides smoother optimization.

---

## Q7 — Is Being Robust to Outliers Always a Good Thing?

Someone says:

> "MAE is better because it ignores outliers."

Do you agree?

### Answer

Not always.

Sometimes outliers represent the most important mistakes.

Examples include:

- Autonomous driving
- Medical diagnosis
- Aircraft control systems
- Fraud detection

In these applications, a single large mistake can have serious consequences.

In such cases, MSE is often preferred because it strongly penalizes large prediction errors.

---

## 💡 Engineer's Insight

Two models can have exactly the same MAE while making completely different kinds of mistakes.

For example:

Model A

```text
Errors = [5, 5, 5, 5, 5]

MAE = 5
```

Model B

```text
Errors = [0, 0, 0, 0, 25]

MAE = 5
```

Although both models have the same MAE, their behavior is very different.

- Model A makes consistent small mistakes.
- Model B is nearly perfect but occasionally makes a very large mistake.

This demonstrates that MAE summarizes the **average magnitude** of errors but does **not** describe the distribution of those errors.

When evaluating machine learning models, engineers often examine both the loss value and the error distribution.

---

# 🎯 Staff Engineer Challenge

You are leading the AI team for a food delivery platform.

The business team tells you:

```text
Most deliveries are within ±3 minutes.

Occasionally, deliveries are delayed due to traffic accidents, heavy rain, vehicle breakdowns, or other unexpected events.
```

Your ML engineer asks:

> "Should we train the delivery time prediction model using MAE or MSE?"

### Discussion

Before choosing a loss function, a Staff Engineer first understands the business objective.

Questions to ask include:

### 1. What Does the Business Care About?

- Is the goal to minimize the average delivery error?
- Or is avoiding extremely late deliveries more important?

The answer determines which loss function better aligns with the business objective.

---

### 2. Are Large Errors Actually Meaningful?

Some large delays may occur because of:

- Traffic accidents
- Extreme weather
- Vehicle breakdowns
- Unexpected road closures

These events may not reflect weaknesses in the prediction model.

If these outliers are unavoidable, allowing them to dominate training may actually reduce overall model performance.

---

### 3. What Happens Most of the Time?

If thousands of deliveries every day are already predicted within a few minutes of the actual delivery time, then optimizing for **average performance** may provide greater business value than focusing on a handful of rare delays.

In such situations, MAE is often a reasonable choice because it is robust to outliers.

---

### 4. When Would MSE Become the Better Choice?

Suppose the company introduces the following policy:

```text
Any delivery delayed by more than 20 minutes

↓

Customer receives a ₹300 refund.
```

Now every large prediction error has a direct financial impact.

In this case, MSE becomes much more attractive because it strongly penalizes large mistakes, encouraging the model to reduce costly prediction failures.

---

## 💡 Staff Engineer Insight

A Staff Engineer does not choose a loss function based only on mathematics.

Instead, they ask:

- What business metric are we optimizing?
- Are outliers meaningful or simply noise?
- What is the cost of a large prediction error?
- Which loss function best aligns with the business objective?

The same dataset may require different loss functions depending entirely on the business goals.

Choosing the right loss function is ultimately an engineering decision, not just a mathematical one.

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