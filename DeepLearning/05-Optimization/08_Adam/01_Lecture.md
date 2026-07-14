# 🚀 Adam Optimizer — Combining the Best of Both Worlds

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to answer:

- ✅ Why Momentum alone is not sufficient
- ✅ Why RMSProp alone is not sufficient
- ✅ What problem Adam tries to solve
- ✅ Why Adam combines Momentum and RMSProp
- ✅ What are the two moving averages maintained by Adam
- ✅ Why Adam became the default optimizer for Deep Learning
- ✅ Why Adam works particularly well for modern LLMs

---

# 🌍 Chapter 1 — The Journey So Far

Let's quickly revisit the evolution of optimizers.

### 🟢 Gradient Descent

We started with vanilla Gradient Descent.

```text
Current Gradient
        ↓
Weight Update
```

Problem:

- ❌ No memory
- ❌ Zig-zag movement
- ❌ Slow convergence

---

### 🟡 Momentum

Momentum asked an important question.

> "Why should every update forget the previous direction?"

Instead of relying only on the current gradient,

it remembers previous movement.

```text
Past Movement
      +
Current Gradient
      ↓
Velocity
      ↓
Weight Update
```

Momentum helped optimization move faster and reduced oscillations.

But another question remained.

---

### 🔵 AdaGrad

Momentum improved **direction**.

But it still used the **same learning rate** for every parameter.

AdaGrad asked:

> "Should every parameter really receive the same learning rate?"

Its answer:

```text
Large Historical Gradient
        ↓
Small Learning Rate

Small Historical Gradient
        ↓
Large Learning Rate
```

This introduced **adaptive learning rates**.

However,

AdaGrad had a serious limitation.

---

### 🔴 RMSProp

AdaGrad remembered **everything**.

```text
All Past Gradients
        ↓
Learning Rate
```

After millions of iterations,

old gradients continued affecting today's learning rate.

Eventually,

the learning rate became extremely small.

Training almost stopped.

RMSProp fixed this by asking:

> "Shouldn't recent gradients matter more than gradients from years ago?"

Instead of storing the complete history,

RMSProp uses an **Exponential Moving Average (EMA)**.

```text
Recent Gradients
      ↑
Higher Weight

Old Gradients
      ↓
Lower Weight
```

This solved AdaGrad's biggest weakness.

---

# 🤔 Chapter 2 — But Are We Finally Done?

At first glance,

Momentum and RMSProp appear to solve different problems.

Momentum gives us

✅ Better direction

RMSProp gives us

✅ Better learning rates

A natural question arises.

> Why not use both together?

Imagine driving a car.

To reach your destination safely,

you need **both**:

🧭 Good steering

🚗 Proper speed

Having only one is insufficient.

Similarly,

an optimizer should ideally know:

- Which direction to move.
- How aggressively to move.

This idea led to the development of **Adam**.

---

# 💡 Chapter 3 — The Core Idea Behind Adam

Adam stands for

> **Adaptive Moment Estimation**

The name itself tells us what it does.

It estimates **two different moments** of the gradients.

Instead of remembering only one thing,

Adam remembers **two kinds of history**.

---

## 📌 First Memory — Momentum

The first memory tracks

```text
Average Gradient
```

using EMA.

```text
m = β₁ × m + (1 − β₁) × g
```

This is exactly what Momentum was doing.

Purpose:

✅ Learn the direction of movement.

---

## 📌 Second Memory — RMSProp

The second memory tracks

```text
Average Squared Gradient
```

using EMA.

```text
v = β₂ × v + (1 − β₂) × g²
```

This is exactly RMSProp.

Purpose:

✅ Adapt the learning rate.

---

# 🧩 Chapter 4 — Two Memories Working Together

Instead of maintaining one historical value,

Adam maintains two.

```text
               Gradient
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼

Average Gradient      Average Squared Gradient

(Momentum)               (RMSProp)

      │                       │
      └───────────┬───────────┘
                  ▼

          Weight Update
```

One decides **where to move**.

The other decides **how aggressively to move**.

---

# 🧮 Chapter 5 — Adam Equations

### Step 1

Update Momentum

```text
m = β₁ × m + (1 − β₁) × g
```

---

### Step 2

Update RMSProp Memory

```text
v = β₂ × v + (1 − β₂) × g²
```

---

### Step 3

Compute Update

```text
Weight Update

= η × m / √v
```

Notice something interesting.

The numerator comes from Momentum.

The denominator comes from RMSProp.

Adam literally combines both optimizers.

---

# 🎯 Chapter 6 — Understanding Each Component

### 🟢 Numerator

```text
m
```

Answers:

> Which direction should I move?

---

### 🔵 Denominator

```text
√v
```

Answers:

> How large should my step be?

---

### 🟣 Learning Rate

```text
η
```

Controls the overall scale.

---

# 🚀 Chapter 7 — Why Adam Became So Popular

Adam combines the strengths of previous optimizers.

| Optimizer | Direction Memory | Adaptive Learning Rate |
|-----------|------------------|------------------------|
| Gradient Descent | ❌ | ❌ |
| Momentum | ✅ | ❌ |
| AdaGrad | ❌ | ✅ |
| RMSProp | ❌ | ✅ |
| Adam | ✅ | ✅ |

Adam usually requires much less tuning.

It works well for

- 🖼️ Computer Vision
- 💬 NLP
- 🤖 Transformers
- 🧠 Large Language Models
- 📈 Large-scale Deep Learning

# ⚠️ Chapter 8 — Is Adam Perfect?

No optimizer is perfect.

Although Adam converges quickly,

research later showed

it sometimes produces poorer generalization

than SGD with Momentum.

This eventually led to

👉 AdamW

which fixes one of Adam's biggest issues.

We'll study that next.

---

# ⚖️ Chapter 9 — The Hidden Problem: Bias in the First Few Iterations

At this point, Adam seems complete.

We maintain two memories:

```text
First Moment  (Momentum)

Second Moment (RMSProp)
```

and combine them to update the weights.

So are we done?

Not quite.

There is a subtle problem that appears **only during the first few training iterations.**

---

## 🤔 Let's Think From First Principles

Recall how both moving averages are initialized.

```text
m = 0

v = 0
```

At the very beginning of training,

Adam has **no history**.

Now suppose the very first gradient is

```text
g = 10
```

Using Momentum,

```text
m = β₁ × 0 + (1 − β₁) × 10
```

If

```text
β₁ = 0.9
```

then

```text
m = 1
```

But wait...

The actual gradient was

```text
10
```

Why is our estimate only

```text
1 ?
```

---

## 🚨 The Moving Average Starts Too Small

The same thing happens with RMSProp.

Initially,

```text
v = 0
```

After the first update,

```text
v = β₂ × 0 + (1 − β₂) × g²
```

Again,

the estimate becomes much smaller than the true value.

Both memories are **biased toward zero** simply because they started from zero.

This has nothing to do with the data.

It is purely an initialization problem.

---

## 🎯 Why Does This Matter?

Remember,

Adam computes

```text
Weight Update

= η × m / √v
```

If both

```text
m

and

v
```

are underestimated,

then the weight update is also inaccurate.

During the first few iterations,

Adam is not using reliable estimates of either

- direction
- or adaptive learning rate.

---

## 💡 The Solution: Bias Correction

Instead of accepting these underestimated values,

Adam corrects them mathematically.

The corrected first moment becomes

```text
m̂ = m / (1 − β₁ᵗ)
```

The corrected second moment becomes

```text
v̂ = v / (1 − β₂ᵗ)
```

where

```text
t
```

is the current iteration number.

These equations compensate for the fact that the moving averages started at zero.

---

## 📈 What Happens Over Time?

During the first few iterations,

```text
Bias Correction

↓

Very Important
```

As training progresses,

the moving averages accumulate sufficient history.

Eventually,

```text
β₁ᵗ → 0

β₂ᵗ → 0
```

so

```text
1 − βᵗ

≈ 1
```

and the correction becomes almost unnecessary.

Bias correction mainly affects the **early stages of training.**

---

# 🤔 Chapter 10 — The Hidden Problem with Adam

At first glance,

Adam appears to be complete.

It combines:

- ✅ Momentum (Direction)
- ✅ RMSProp (Adaptive Learning Rate)

So why wasn't Adam immediately finalized after combining these two ideas?

The answer lies in a subtle issue introduced by **Exponential Moving Average (EMA)** itself.

Both Momentum and RMSProp maintain running averages.

For Momentum,

```text
m = β₁ × m + (1 − β₁) × g
```

For RMSProp,

```text
v = β₂ × v + (1 − β₂) × g²
```

But before training begins,

Adam has never seen any gradients.

So it initializes both memories as

```text
m = 0

v = 0
```

This seems harmless.

However,

this simple initialization introduces an unexpected bias during the first few iterations.

---

# 🚀 Chapter 11 — Understanding Bias Correction

## 📌 First Iteration

Suppose

```text
β₁ = 0.9

Gradient = 10
```

Initially,

```text
m = 0
```

Applying the Momentum update,

```text
m = 0.9 × 0 + 0.1 × 10

m = 1
```

Adam now believes

```text
Average Gradient = 1
```

Pause for a moment.

How many gradients have we actually observed?

Only one.

That gradient was

```text
10
```

So shouldn't our estimate also be

```text
10 ?
```

Why is Adam saying

```text
1 ?
```

---

## 💡 Where Did This Error Come From?

The problem is that Adam is still influenced by the artificial initialization.

Conceptually,

it behaves as if the history were

```text
0

10
```

instead of simply

```text
10
```

That fake zero pulls the moving average downward.

This phenomenon is called

> **Initialization Bias**

---

## 📌 Second Iteration

Suppose the next gradient is

```text
20
```

Now

```text
m = 0.9 × 1 + 0.1 × 20

m = 2.9
```

Adam estimates

```text
Average Gradient = 2.9
```

Reality

We have observed

```text
10

20
```

whose average is

```text
15
```

Again,

Adam underestimates the true average.

---

## 📌 Third Iteration

Suppose

```text
Gradient = 30
```

Then

```text
m = 0.9 × 2.9 + 0.1 × 30

m = 5.61
```

Reality

```text
Observed Gradients

10

20

30

Average = 20
```

Adam still estimates

```text
5.61
```

Although the estimate is improving,

it remains heavily biased downward because of the initial zero.

---

# 📊 Comparing EMA vs Actual Average

| Iteration | Gradient | EMA (m) | Actual Average |
|-----------|----------|---------|----------------|
| 1 | 10 | 1.00 | 10 |
| 2 | 20 | 2.90 | 15 |
| 3 | 30 | 5.61 | 20 |

Notice the pattern.

EMA is consistently smaller.

Not because EMA is incorrect,

but because it started from zero.

---

# 🎓 Real-Life Analogy

Imagine a teacher calculating the average marks of students.

The first three students score

```text
90

80

70
```

The average should be

```text
80
```

Now imagine the teacher accidentally includes one imaginary student who scored

```text
0
```

The average becomes

```text
0

90

80

70

Average = 60
```

That fake zero unfairly lowers the average.

Exactly the same thing happens inside Adam.

---

# 🤔 How Do We Remove This Fake Zero?

Researchers realized that the influence of the initial zero can be calculated mathematically.

The correction is

```text
m̂ = m / (1 − β₁ᵗ)
```

Similarly,

```text
v̂ = v / (1 − β₂ᵗ)
```

This process is called **Bias Correction**.

---

# 🚀 Applying Bias Correction

### First Iteration

Before correction

```text
m = 1
```

Applying the correction

```text
m̂ = 1 / (1 − 0.9¹)

= 1 / 0.1

= 10
```

Perfect!

The corrected estimate now matches the observed gradient.

---

### Second Iteration

Before correction

```text
m = 2.9
```

Applying the correction

```text
m̂ = 2.9 / (1 − 0.9²)

= 2.9 / 0.19

≈ 15.26
```

Very close to the true average

```text
15
```

---

### Third Iteration

Before correction

```text
m = 5.61
```

Applying the correction

```text
m̂ = 5.61 / (1 − 0.9³)

= 5.61 / 0.271

≈ 20.70
```

Again,

very close to the true average

```text
20
```

---

# 📈 Why Does Bias Correction Disappear Later?

Notice the denominator

```text
1 − β₁ᵗ
```

Initially,

```text
t = 1

1 − 0.9 = 0.1
```

After many iterations,

```text
0.9¹⁰⁰ ≈ 0
```

Therefore,

```text
1 − 0.9¹⁰⁰ ≈ 1
```

which means

```text
m̂ ≈ m
```

The correction naturally becomes unnecessary because the influence of the initial zero vanishes over time.

---

# 🧮 Chapter 12 — The Complete Adam Algorithm

Now we have everything required to understand Adam.

### Step 1 — Compute the Gradient

```text
g = ∇L
```

---

### Step 2 — Update the First Moment (Momentum)

```text
m = β₁ × m + (1 − β₁) × g
```

---

### Step 3 — Update the Second Moment (RMSProp)

```text
v = β₂ × v + (1 − β₂) × g²
```

---

### Step 4 — Apply Bias Correction

```text
m̂ = m / (1 − β₁ᵗ)

v̂ = v / (1 − β₂ᵗ)
```

---

### Step 5 — Update the Weights

```text
W(new) = W(old) − η × m̂ / (√v̂ + ε)
```

where

- `η` → Learning Rate
- `m̂` → Bias-corrected first moment
- `v̂` → Bias-corrected second moment
- `ε` → Small constant for numerical stability

---

# 🏗️ Engineer's Insight

Think of Bias Correction as removing the influence of an imaginary history.

Without Bias Correction,

Adam behaves as though it had already seen many zero gradients before training even began.

Bias Correction simply removes this artificial bias,

allowing the optimizer to make accurate updates from the very first iteration.

---

# 📌 Key Takeaways

```text
🧠 Adam initializes both moving averages to zero.

⚠️ This causes the first few EMA estimates to be artificially small.

📉 The smaller estimates slow down early learning.

✨ Bias Correction mathematically removes the influence of the initial zero.

🚀 After many iterations, the correction naturally becomes negligible.

🎯 Final Adam = Momentum + RMSProp + Bias Correction.
```
---

## 🧠 Engineer's Insight

Imagine asking five employees to estimate the average monthly sales of a company.

One employee has worked there for

```text
5 years
```

Another joined

```text
yesterday.
```

Would you trust both averages equally?

Of course not.

The new employee simply hasn't seen enough data.

Adam faces the same problem.

During the first few iterations,

its moving averages have seen very little history,

so they underestimate the true statistics.

Bias correction compensates for this lack of historical information until sufficient observations have been collected.

---

## 📌 Chapter Summary

```text
🚨 EMA starts from zero.

📉 Early moving averages underestimate reality.

⚖️ This creates biased estimates of both momentum and variance.

🛠️ Adam applies bias correction.

📈 As iterations increase,

Bias Correction gradually becomes unnecessary.

🎯 Final Adam uses corrected moments, not raw moments.
```

---

## 🔜 Coming Next

Now that Adam is mathematically complete,

we'll study its practical limitation.

> 🤔 If Adam works so well...

why was **AdamW** invented?

We'll discover how **weight decay** interacts with Adam and why decoupling it led to better generalization in modern deep learning.

---

# 🏗️ Engineer's Insight

Think of Adam as driving with both

🧭 GPS

and

🚗 Cruise Control.

The GPS tells you

where to go.

Cruise Control decides

how fast you should travel.

Removing either one makes driving less effective.

Adam simply combines both.

---

# 📌 Key Takeaways
```text
🎯 Momentum remembers previous movement.
📉 RMSProp remembers recent squared gradients.
🧠 Adam combines both ideas.
⚡ Adam maintains two separate memories:
     ✅First Moment → Momentum
     ✅Second Moment → RMSProp
🚀 Adam adapts both direction and learning rate.
🌍 Adam is one of the most widely used optimizers in Deep Learning.
➡️ Adam is not perfect; AdamW later improves upon it.
```
---