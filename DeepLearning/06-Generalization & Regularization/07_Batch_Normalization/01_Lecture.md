# 📘 Lecture 7 — Batch Normalization

---

# 🎯 Learning Objectives

By the end of this lecture, you will understand:

- ✅ Why Batch Normalization was introduced
- ✅ What Internal Covariate Shift means
- ✅ Why deep neural networks become difficult to train
- ✅ How Batch Normalization works step-by-step
- ✅ Why every mini-batch is normalized
- ✅ The intuition behind mean, variance, normalization, γ (Gamma), and β (Beta)

---

# 📖 Chapter 1 — Why was Batch Normalization Introduced?

Before Batch Normalization was introduced, researchers noticed something interesting.

As neural networks became deeper,

training became increasingly difficult.

Instead of learning faster,

the networks often became:

- ❌ Slow to converge
- ❌ Difficult to optimize
- ❌ Sensitive to initialization
- ❌ Sensitive to learning rate
- ❌ Unstable during training

Surprisingly,

simply adding more layers did **not** always improve performance.

Sometimes,

a deeper network performed **worse** than a shallower one.

---

## 🤔 Why does this happen?

Remember how a neural network learns.

```text
Input
   ↓
Layer 1
   ↓
Layer 2
   ↓
Layer 3
   ↓
Layer 4
   ↓
Prediction
```

Every layer receives the output of the previous layer.

If the output of one layer keeps changing,

then every following layer also receives changing inputs.

As a result,

each layer has to continuously adapt to a moving target.

Training therefore becomes much harder.

---

## 💡 Intuition

Imagine you're preparing for an exam.

Every day,

your teacher changes the syllabus.

Monday

```text
Study Chapters 1–5
```

Tuesday

```text
Ignore Chapter 3.
Add Chapters 6–7.
```

Wednesday

```text
Chapter 2 is removed.
```

Thursday

```text
Everything changes again.
```

Would learning become easier?

Obviously not.

You would spend most of your time adapting,

instead of actually learning.

Exactly the same thing happens inside a deep neural network.

Each layer keeps seeing changing input distributions.

---

# 📖 Chapter 2 — Internal Covariate Shift

This changing input distribution has a name.

It is called

# 📌 Internal Covariate Shift

---

## 📖 Definition

Internal Covariate Shift means:

> The distribution of inputs to a layer keeps changing during training because the previous layers are continuously updating their weights.

Every gradient update changes the weights.

Updated weights produce different activations.

Different activations become new inputs for the next layer.

Therefore,

every layer keeps learning from a constantly changing distribution.

---

## 🏠 Real-Life Analogy

Imagine a cricket coach training a batsman.

Every day,

the bowling machine changes.

Day 1

```text
Speed = 120 km/h
```

Day 2

```text
Speed = 145 km/h
```

Day 3

```text
Speed = 90 km/h
```

Day 4

```text
Ball swings differently.
```

The batsman now spends more time adapting to changing conditions

instead of improving technique.

Deep neural networks face the same problem.

---

## 📊 Visualizing Internal Covariate Shift

Suppose Layer 3 expects inputs roughly like this:

```text
Mean ≈ 0

Variance ≈ 1
```

Iteration 1

```text
Mean = 0.2

Variance = 1.1
```

Iteration 2

```text
Mean = 3.8

Variance = 5.2
```

Iteration 3

```text
Mean = -2.1

Variance = 0.4
```

Iteration 4

```text
Mean = 1.7

Variance = 8.6
```

Every optimization step changes the statistics.

Layer 3 must constantly relearn how to process its inputs.

---

# 📖 Chapter 3 — The Core Idea Behind Batch Normalization

The idea is surprisingly simple.

Instead of allowing every layer to receive inputs with arbitrary scales,

we first normalize those inputs.

```text
Layer Output
       ↓
Normalize
       ↓
Pass to Next Layer
```

Now every layer receives inputs that are much more stable.

Instead of chasing a moving target,

each layer learns from a much more consistent distribution.

Training becomes faster and more stable.

---

## 🎯 Goal of Batch Normalization

Batch Normalization tries to make the inputs to every layer approximately:

```text
Mean = 0

Variance = 1
```

This standardized distribution makes optimization much easier.

---

# 📖 Chapter 4 — Why Normalize Every Mini-Batch?

A natural question is:

> Why not normalize the entire dataset once?

Because neural network activations are **not fixed**.

Remember,

after every optimization step,

the weights change.

```text
Weights Updated
        ↓
Activations Change
        ↓
Input Distribution Changes
```

Even if the original dataset never changes,

the outputs produced by hidden layers change continuously.

Therefore,

normalization must happen repeatedly during training.

---

## 📦 Why Mini-Batches?

Modern deep learning rarely trains on the entire dataset at once.

Instead,

training happens like this:

```text
Dataset

↓

Mini Batch 1

↓

Mini Batch 2

↓

Mini Batch 3

↓

Mini Batch 4
```

For every mini-batch,

Batch Normalization computes fresh statistics.

Each mini-batch gets normalized independently.

---

## 📌 Why not use the whole dataset?

Using the full dataset every iteration would be:

- ❌ Extremely slow
- ❌ Memory intensive
- ❌ Impractical for large datasets

Mini-batches provide a fast approximation.

---

# 📖 Chapter 5 — Batch Normalization Pipeline

Every forward pass follows the same sequence.

```text
Input Activations
        ↓
Compute Batch Mean (μ)
        ↓
Compute Batch Variance (σ²)
        ↓
Normalize Activations
        ↓
Scale using γ
        ↓
Shift using β
        ↓
Pass to Activation Function
```

This entire process happens automatically for every mini-batch during training.


# 📖 Chapter 6 — Mathematical Steps of Batch Normalization

Now that we understand **why** Batch Normalization was introduced,

let's see **how** it works mathematically.

For every mini-batch, BatchNorm performs **five steps**.

```text
Input Activations
        │
        ▼
Compute Mean (μ)
        │
        ▼
Compute Variance (σ²)
        │
        ▼
Normalize
        │
        ▼
Scale (γ)
        │
        ▼
Shift (β)
        │
        ▼
Output Activations
```

---

## 🟢 Step 1 — Compute Batch Mean

Suppose a mini-batch contains four activation values.

```text
2
4
6
8
```

The batch mean is

```text
μ = (2 + 4 + 6 + 8) / 4

μ = 5
```

The mean represents the center of the batch.

---

## 🟢 Step 2 — Compute Batch Variance

Variance tells us how spread out the activations are.

```text
σ² = Average[(x − μ)²]
```

For our example,

```text
Values

2
4
6
8

Mean = 5
```

Differences

```text
-3
-1
 1
 3
```

Squares

```text
9
1
1
9
```

Average

```text
σ² = (9+1+1+9)/4

σ² = 5
```

---

## 🟢 Step 3 — Normalize

Now every activation is converted into a standardized value.

The normalization equation is

```text
x̂ = (x − μ)
/ √(σ² + ε)
```

where

```text
μ = Batch Mean

σ² = Batch Variance

ε = Small number for numerical stability
```

---

### 🤔 Why do we add ε (epsilon)?

Imagine all activations are identical.

```text
5
5
5
5
```

Then

```text
Variance = 0
```

The denominator becomes

```text
√0
```

which causes division by zero.

To avoid this,

BatchNorm adds a very tiny value.

```text
ε = 0.00001
```

This has almost no effect on the result,

but prevents numerical errors.

---

# 📖 Chapter 7 — Why are γ (Gamma) and β (Beta) Needed?

After normalization,

every activation has approximately

```text
Mean = 0

Variance = 1
```

At first glance,

this sounds perfect.

But it creates a new problem.

---

## 🤔 Problem

What if a layer actually needs

```text
Mean = 8

Variance = 4
```

for the task?

BatchNorm has forced everything to

```text
Mean = 0

Variance = 1
```

The network has now lost its flexibility.

---

## 💡 Solution

BatchNorm introduces two learnable parameters.

```text
γ (Gamma)

β (Beta)
```

The final output becomes

```text
y = γx̂ + β
```

---

### 🎯 Role of Gamma (γ)

Gamma controls the scale.

```text
Large γ

↓

Larger activations
```

```text
Small γ

↓

Smaller activations
```

---

### 🎯 Role of Beta (β)

Beta shifts the activations.

```text
Positive β

↓

Move distribution right
```

```text
Negative β

↓

Move distribution left
```
Batch Normalization first normalizes activations to stabilize training, then uses two learnable parameters, γ (scale) and β (shift), to restore the model's flexibility. This allows the network to learn the most suitable activation distribution for the task instead of being permanently constrained to mean 0 and variance 1.
---

## 📦 Complete BatchNorm Equation

```text
Mean

↓

Variance

↓

Normalize

↓

Scale (γ)

↓

Shift (β)
```

Mathematically,

```text
x̂ = (x − μ)
/ √(σ² + ε)

y = γx̂ + β
```

Notice that **γ** and **β** are learned automatically during training.

---

# 📖 Chapter 8 — Training vs Inference

One of the most common interview questions is:

> Does BatchNorm behave the same during training and inference?

The answer is **No.**

---

## 🟢 During Training

Every mini-batch has different statistics.

For every mini-batch,

BatchNorm computes

- Mean
- Variance

from that batch only.

```text
Mini Batch

↓

Compute Mean

↓

Compute Variance

↓

Normalize
```

---

## 🟢 During Inference

During inference,

we process

```text
One image

One sentence

One sample
```

There is no meaningful mini-batch.

So where do the mean and variance come from?

---

## 💡 Running Statistics

During training,

BatchNorm continuously stores

```text
Running Mean

Running Variance
```

These are moving averages collected across many mini-batches.

During inference,

BatchNorm uses these stored values.

```text
Running Mean

Running Variance

↓

Normalize
```

---

## 📌 Why?

Because predictions must be deterministic.

The same input should always produce the same output.

If BatchNorm computed statistics from every inference sample,

predictions would become inconsistent.

---

# 📖 Chapter 9 — Benefits of Batch Normalization

BatchNorm became one of the biggest breakthroughs in deep learning because it provides many benefits.

---

## 🚀 1. Faster Training

Stable activations allow gradients to propagate more efficiently.

Training converges much faster.

---

## 🚀 2. Higher Learning Rates

Without BatchNorm,

large learning rates often cause divergence.

BatchNorm stabilizes training,

allowing much larger learning rates.

---

## 🚀 3. Less Sensitive to Initialization

Without BatchNorm,

poor weight initialization can completely break training.

BatchNorm reduces this sensitivity.

---

## 🚀 4. Reduced Internal Covariate Shift

Each layer receives much more stable input distributions.

Optimization becomes easier.

---

## 🚀 5. Mild Regularization

Every mini-batch has slightly different statistics.

This introduces a small amount of randomness.

As a result,

BatchNorm provides a mild regularization effect,

which can slightly reduce overfitting.

However,

it is **not** a replacement for Dropout.

---

# 📖 Chapter 10 — BatchNorm vs Dropout

Many beginners think these are the same.

They are not.

| Feature | BatchNorm | Dropout |
|----------|-----------|----------|
| 🎯 Primary Goal | Stabilize training | Reduce overfitting |
| 🧠 Main Idea | Normalize activations | Randomly drop neurons |
| ⚡ Speeds up training | ✅ Yes | ❌ No |
| 🛡️ Reduces overfitting | Slightly | Strongly |
| 🔄 Uses mini-batch statistics | ✅ Yes | ❌ No |
| 🚫 Removes neurons | ❌ No | ✅ Yes |

---

## 🎯 Simple Analogy

### 📦 BatchNorm

Imagine every student receives textbooks printed in the same font size.

Everyone starts learning from a standardized format.

---

### 🎲 Dropout

Imagine random students are absent every day.

Everyone has to learn independently without depending on particular classmates.

---

# 📖 Chapter 11 — Limitations of Batch Normalization

Although BatchNorm is extremely powerful,

it is not perfect.

---

## ❌ 1. Depends on Batch Size

Very small mini-batches produce poor estimates of

- Mean
- Variance

Training may become unstable.

---

## ❌ 2. Extra Computation

Every forward pass computes

- Mean
- Variance
- Normalization

which adds a small computational overhead.

---

## ❌ 3. Not Ideal for Sequential Models

In models like

- RNN
- LSTM

batch statistics may vary significantly across time steps.

This is why alternatives like **Layer Normalization** are commonly used in Transformers and modern LLMs.

> 📌 We'll study Layer Normalization in a later lecture.

---

# 🎯 Summary

```text
Deep Network

↓

Internal Covariate Shift

↓

Changing Activation Distribution

↓

Batch Normalization

↓

Compute Mean

↓

Compute Variance

↓

Normalize

↓

Scale (γ)

↓

Shift (β)

↓

Stable Activations

↓

Faster Training
```

---

# 💡 Key Takeaways

✅ BatchNorm normalizes activations for every mini-batch.

✅ It computes batch mean and variance during training.

✅ It uses running mean and running variance during inference.

✅ Gamma (γ) restores the model's flexibility by scaling activations.

✅ Beta (β) shifts the normalized activations.

✅ BatchNorm speeds up convergence and allows higher learning rates.

✅ It provides a mild regularization effect but does **not** replace Dropout.

✅ BatchNorm and Dropout solve different problems and are often used for different purposes.

---


