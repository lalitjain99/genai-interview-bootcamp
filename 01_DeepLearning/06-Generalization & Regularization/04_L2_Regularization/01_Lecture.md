# 📘L2 Regularization
## Shrinking Weights Instead of Removing Them

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to answer:

- Why wasn't L1 Regularization sufficient for every problem?
- What motivated researchers to invent L2 Regularization?
- What is the intuition behind L2 Regularization?
- Why are weights squared instead of taking their absolute values?
- How does L2 differ from L1?
- Why doesn't L2 perform feature selection?
- Why is L2 also called Weight Decay or Ridge Regularization?

---

# 📚 Chapter 1 — Did L1 Solve Everything?

In the previous lecture, we learned that **L1 Regularization** helps reduce overfitting.

It penalizes large weights.

As a result,

unimportant features gradually receive **zero weights**.

The model automatically performs **feature selection**.

Researchers were excited.

It seemed like the perfect solution.

But after applying L1 to many real-world datasets,

they noticed something interesting.

Sometimes...

removing features completely wasn't the best idea.

---

# 📚 Chapter 2 — When Every Feature Matters

Imagine you're building a **House Price Prediction** model.

Your features are

```text
Area

Bedrooms

Location

Parking

Age of House
```

Notice something.

Every feature contributes something to the prediction.

Some contribute more.

Some contribute less.

But none of them are completely useless.

Now suppose L1 Regularization is applied.

As training progresses,

some weights may become exactly zero.

```text
Area        = 4.3

Bedrooms    = 2.1

Location    = 4.8

Parking     = 0

Age         = 1.4
```

The model has completely removed the **Parking** feature.

But...

is Parking really useless?

Not necessarily.

Perhaps it contributes only a little,

but that little information still improves predictions.

Researchers began asking a different question.

---

# 💡 Core Idea

L1 asks a simple question.

> **"Can I remove this feature?"**

Researchers now wondered,

> **"Instead of deleting features completely...**

> **can we simply make every weight smaller?"**

In other words,

instead of encouraging **feature elimination**,

what if we encouraged **weight moderation**?

This small shift in thinking gave birth to another regularization technique.

**L2 Regularization.**

---

# 📚 Chapter 3 — The Philosophy Behind L2

Think of two teachers.

### Teacher A (L1)

Says

```text
This topic isn't important.

Remove it completely.
```

### Teacher B (L2)

Says

```text
Everything is important.

But don't spend all your time on one topic.
```

L2 follows Teacher B's philosophy.

It doesn't want to eliminate features.

It wants every feature to contribute,

but in a balanced way.

---

# 📌 Chapter 4 — The L2 Penalty

Unlike L1,

which adds the **absolute value** of every weight,

L2 adds the **square** of every weight.

Mathematically,

```text
L2 Regularization Loss

=

w₁²

+

w₂²

+

...

+

wₙ²
```

The Total Loss becomes

```text
Total Loss

=

Prediction Loss

+

λ × L2 Regularization Loss
```

Notice something important.

The **Prediction Loss** remains exactly the same.

Only the **Regularization Loss** has changed.

The optimizer now tries to minimize both

- Prediction Error
- Model Complexity

simultaneously.

---

# 🧠 Intuition — Why Square the Weights?

At first glance,

squaring the weights may look like an arbitrary mathematical choice.

It isn't.

Researchers deliberately chose squaring because it penalizes **large weights much more aggressively** than small ones.

Consider two weights.

```text
Weight A = 1

Weight B = 10
```

With **L1 Regularization**

```text
Penalty

Weight A → 1

Weight B → 10
```

Now look at **L2 Regularization**

```text
Penalty

Weight A → 1² = 1

Weight B → 10² = 100
```

Notice the difference.

The larger weight became **ten times larger**...

but its penalty became **one hundred times larger**.

Large weights become extremely expensive.

Small weights receive only a mild penalty.

This is exactly what researchers wanted.

---

# 📊 Visual Comparison

| Weight | L1 Penalty | L2 Penalty |
|---------|-----------:|-----------:|
| 0.5 | 0.5 | 0.25 |
| 1 | 1 | 1 |
| 2 | 2 | 4 |
| 5 | 5 | 25 |
| 10 | 10 | 100 |

💡 Observe the pattern.

As the weight grows,

the L2 penalty grows **much faster** than the L1 penalty.

---

# 💼 Engineering Insight

Remember,

L2 is **not trying to eliminate features**.

Instead,

it discourages any single feature from dominating the prediction.

This often leads to

- smoother models
- better generalization
- more stable predictions

Especially when every feature carries some useful information.

---

# 📚 Chapter 6 — How Does L2 Change the Optimizer?

Let's revisit our optimizer.

Just like L1,

the optimizer is still trying to minimize

```text
Total Loss

=

Prediction Loss

+

λ × L2 Regularization Loss
```

Nothing has changed here.

The only difference is

**how the Regularization Loss behaves.**

---

Suppose our model has learned the following weights.

```text
Area         = 4.8

Bedrooms     = 2.4

Location     = 5.3

Parking      = 0.8

Age          = 1.2
```

L2 doesn't ask

> "Can I remove Parking?"

Instead it asks

> "Can I reduce every weight a little...

without hurting the Prediction Loss too much?"

The optimizer therefore starts shrinking every weight.

```text
Before L2

Area         = 4.8

Bedrooms     = 2.4

Location     = 5.3

Parking      = 0.8

Age          = 1.2
```

↓

```text
After L2

Area         = 4.2

Bedrooms     = 2.1

Location     = 4.7

Parking      = 0.6

Age          = 1.0
```

Notice something.

No feature disappeared.

Every feature still contributes.

Only their influence has been moderated.

---

# 💡 Core Idea

L1 says

```text
Some features

↓

Completely disappear.
```

L2 says

```text
Every feature stays.

Nobody gets too much influence.
```

That is the entire philosophy of L2.

---

# 📚 Chapter 7 — Why Doesn't L2 Produce Zero Weights?

This is one of the most common interview questions.

Many students assume

> "Since L2 penalizes weights...

eventually every weight should become zero."

Interestingly,

that almost never happens.

Why?

Because the optimizer is balancing two competing objectives.

```text
Prediction Loss

↓

Keep useful weights.


Regularization

↓

Shrink every weight.
```

As a weight becomes smaller,

reducing it further provides only a tiny improvement in the Regularization Loss,

while it may noticeably increase the Prediction Loss.

Eventually,

the optimizer reaches a balance.

Instead of

```text
0
```

the weight becomes

```text
0.73

0.18

1.42
```

Small...

but not zero.

This is why L2 produces **dense models** instead of sparse ones.

---

# 📊 L1 vs L2

| Property | L1 Regularization | L2 Regularization |
|-----------|-------------------|-------------------|
| Penalty | \|w\| | w² |
| Removes Features | ✅ Yes | ❌ No |
| Produces Sparse Models | ✅ Yes | ❌ No |
| Shrinks Large Weights | Moderate | Strong |
| Feature Selection | Automatic | No |
| Correlated Features | Keeps one, removes others | Keeps both with smaller weights |

---

# 🧠 Intuition — Balloon Analogy

Imagine five balloons.

Each balloon represents a feature weight.

### L1

Walks around with a pin.

```text
🎈

🎈

🎈

🎈

🎈
```

Some balloons are popped completely.

```text
💥

🎈

💥

🎈

🎈
```

Those features disappear.

---

Now imagine L2.

Instead of carrying a pin,

it opens the valve of every balloon.

```text
🎈

↓

🎈

↓

🎈
```

Every balloon loses a little air.

None disappear.

Some shrink more.

Some shrink less.

This is exactly how L2 behaves.

---

# 📚 Chapter 8 — Why Is L2 Called Weight Decay?

In Machine Learning,

L2 Regularization is also known as

- Ridge Regularization
- Weight Decay

The term **Weight Decay** comes from what happens during training.

Every optimization step gently pulls the weights toward zero.

Large weights gradually "decay."

Notice,

they don't suddenly vanish.

They slowly become smaller over many iterations.

Hence the name

**Weight Decay.**

---

# 💼 Engineering Perspective

If you've used

PyTorch

or

TensorFlow,

you've probably seen

```python
weight_decay = 0.01
```

Many engineers think

> "Weight Decay is a completely different technique."

Not really.

For classical optimizers like **SGD**,

Weight Decay and L2 Regularization are mathematically equivalent.

⚠️ **Important**

For modern optimizers such as **AdamW**,

Weight Decay is implemented differently (decoupled from the loss function),

which improves optimization.

Since we've already studied **AdamW**,

you can think of it as

> "Adam + Proper Weight Decay."

We'll revisit this connection when discussing optimizer implementations.

---

# 🚀 Interview Insight

If an interviewer asks

> "Why doesn't L2 perform feature selection?"

A strong answer is

> "Because L2 penalizes the square of every weight, encouraging all weights to become smaller rather than exactly zero. The optimizer therefore retains most features while reducing their influence."

---

# 📚 Chapter 9 — When Should We Use L1 vs L2?

Use **L1** when

✅ You have many irrelevant features.

✅ You want automatic feature selection.

✅ Model interpretability is important.

---

Use **L2** when

✅ Most features contain useful information.

✅ You want smoother models.

✅ You want to reduce overfitting without removing features.

This is why

L2 is the default choice in many Deep Learning models.

---

# ✅ Key Takeaways

- L2 penalizes the square of every weight.
- Large weights receive much larger penalties than small weights.
- L2 shrinks weights instead of removing them.
- L2 usually produces dense models.
- L2 is also called Ridge Regularization or Weight Decay.
- Weight Decay gently pulls weights toward zero during training.
- L1 performs feature selection, while L2 focuses on weight moderation.

---

---

# 🧠 Deep Dive — How Does the Optimizer Balance Prediction Loss and Regularization Loss?

At this point, an important question naturally arises.

The optimizer is trying to minimize two different objectives.

```text
1. Prediction Loss

↓

Reduce prediction error.

2. Regularization Loss

↓

Keep the model simple by shrinking the weights.
```

But these two objectives often compete with each other.

As a weight increases,

Prediction Loss usually decreases.

At the same time,

Regularization Loss increases.

So how does the optimizer decide

whether to increase the weight

or decrease it?

---

## 💡 The Secret

The optimizer is **not** minimizing two separate loss functions.

Instead,

it minimizes a **single objective**.

```text
Total Loss

=

Prediction Loss

+

λ × Regularization Loss
```

Notice carefully.

There is only **one loss function**.

The optimizer never asks

> "Should I minimize Prediction Loss?"

or

> "Should I minimize Regularization Loss?"

It only asks

> "Does this weight update reduce the Total Loss?"

---

## 📊 Example

Suppose we have

```text
Prediction Loss = 20

Regularization Loss = 2

λ = 0.1
```

The Total Loss becomes

```text
20 + (0.1 × 2)

=

20.2
```

---

Now imagine increasing a weight.

Prediction improves.

```text
Prediction Loss

20

↓

17
```

But the Regularization Loss increases.

```text
Regularization Loss

2

↓

20
```

The new Total Loss becomes

```text
17 + (0.1 × 20)

=

19
```

Although the Regularization Loss became much larger,

the **Total Loss decreased**.

The optimizer accepts this update.

---

Now increase the same weight even further.

Prediction improves only slightly.

```text
Prediction Loss

17

↓

16
```

But the Regularization Loss explodes.

```text
Regularization Loss

20

↓

120
```

The Total Loss now becomes

```text
16 + (0.1 × 120)

=

28
```

Now the Total Loss has increased.

The optimizer rejects this direction.

---

# 🧠 Intuition

Imagine two managers reviewing every weight.

### 📈 Prediction Manager

Says

```text
I don't care how large the weight becomes.

If prediction improves,

increase it.
```

---

### 📉 Regularization Manager

Says

```text
I don't care about prediction.

Keep every weight as small as possible.
```

These two managers constantly disagree.

The optimizer listens to both.

The final decision is the one that minimizes the **Total Loss**.

---

# 📌 What Happens During Gradient Descent?

The optimizer doesn't compare the losses directly.

Instead,

it compares their **gradients**.

Mathematically,

```text
∂(Total Loss)/∂w

=

∂(Prediction Loss)/∂w

+

λ × ∂(Regularization Loss)/∂w
```

Notice what each term is saying.

### Prediction Gradient

```text
Increase this weight.

It reduces prediction error.
```

### Regularization Gradient

```text
Decrease this weight.

The model is becoming too complex.
```

The optimizer simply **adds these two gradients together**.

If the combined gradient is

```text
Positive
```

the weight decreases.

If the combined gradient is

```text
Negative
```

the weight increases.

If the combined gradient becomes

```text
0
```

the optimizer has reached the optimal value for that weight.

---

# 💼 Engineering Perspective

This is one of the most important ideas in Deep Learning.

Regularization does **not** directly change the weights.

Instead,

it changes the **gradient** used during optimization.

That is why Regularization and Gradient Descent are deeply connected.

You can think of Regularization as an additional force that constantly pulls the weights toward smaller values,

while the Prediction Loss pushes them toward values that best fit the training data.

The optimizer finds the balance between these two competing forces.

---

# 🚀 Interview Insight

If an interviewer asks,

> **"How does the optimizer balance Prediction Loss and Regularization Loss?"**

A strong answer is:

> "The optimizer does not minimize the two losses independently. It minimizes the Total Loss, which is the sum of Prediction Loss and λ times the Regularization Loss. During backpropagation, the gradients from both terms are added together. The Prediction Gradient tries to improve model accuracy, while the Regularization Gradient discourages large weights. The optimizer updates each weight based on this combined gradient, naturally finding a balance between model accuracy and model complexity."


### 📊 Visualizing the Two Competing Forces

Think of every weight in the neural network as being pulled by **two opposing forces**.

```text
                   Prediction Gradient
          (Reduce Prediction Loss)
                        ➜ ➜ ➜

Weight ●────────────────────────────────────────

                        ⬅ ⬅ ⬅
             Regularization Gradient
      (Keep the Weight Small)
```

The optimizer combines both forces before updating the weight.

- 📈 **Prediction Gradient** tries to increase or decrease the weight so that the model predicts the training data more accurately.
- 📉 **Regularization Gradient** constantly pulls the weight toward zero to keep the model simple and prevent overfitting.

The final update is based on the **combined effect** of these two forces.

```text
                Prediction Gradient
                        ➜ ➜ ➜

Weight ●────────────────────────────────────────

                        ⬅ ⬅
             Regularization Gradient

───────────────────────────────────────────────

Net Result : ➜
Weight increases slightly.
```

If the prediction gradient is stronger than the regularization gradient, the weight increases.

---

```text
                Prediction Gradient
                        ➜ ➜

Weight ●────────────────────────────────────────

                        ⬅ ⬅ ⬅ ⬅
             Regularization Gradient

───────────────────────────────────────────────

Net Result : ⬅
Weight decreases.
```

If the regularization gradient is stronger, the weight decreases.

---

```text
                Prediction Gradient
                        ➜ ➜ ➜

Weight ●────────────────────────────────────────

                        ⬅ ⬅ ⬅
             Regularization Gradient

───────────────────────────────────────────────

Net Result : •
No update.
```

When both forces are equal, the combined gradient becomes approximately zero.

At this point, the optimizer has found the **optimal value** for that weight.

> 💡 **Intuition:** Training a neural network is like a tug of war. One team (Prediction Loss) wants larger or smaller weights to improve accuracy, while the other team (Regularization) wants to keep weights small. The optimizer updates each weight based on the net force created by these two competing objectives.

# 📖 Coming Next

We've now studied the two most important regularization techniques.

L1 asks

> "Should this feature exist?"

L2 asks

> "How important should every feature be?"

The next natural question is

> **Can we combine the strengths of both?**

That idea leads us to **Elastic Net Regularization**.