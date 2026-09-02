# RMSProp — Learning to Forget

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- Why does AdaGrad eventually stop learning?
- Why is remembering every historical gradient not always desirable?
- What properties should a better adaptive optimizer have?
- How can we mathematically make old gradients gradually lose importance?
- What is an Exponential Moving Average (EMA)?
- How does RMSProp solve AdaGrad's biggest limitation?
- Why is RMSProp better suited for long training runs?

---

# Chapter 1 — AdaGrad Solved One Problem but Created Another

In the previous lecture,

we learned AdaGrad.

Its main idea was simple.

Instead of using one fixed learning rate for every parameter,

each parameter should have its own learning rate,

based on the history of gradients it has received.

The accumulated gradient history is

```text
G = G + g²
```

and the effective learning rate becomes

```text
η_eff = η / √G
```

Initially,

this works extremely well.

Parameters receiving large gradients automatically slow down,

while parameters receiving small gradients continue learning.

It appears to solve the problem perfectly.

---

# Chapter 2 — The Hidden Problem

Now imagine training a neural network

for several million iterations.

Ask yourself a simple question.

What happens to

```text
G
```

after every iteration?

It only increases.

```text
Iteration 1

↓

G increases

↓

Iteration 2

↓

G increases again

↓

Iteration 3

↓

Still increasing
```

It never decreases.

It never forgets.

This creates a serious problem.

As

```text
G
```

becomes larger and larger,

the effective learning rate

```text
η_eff = η / √G
```

becomes smaller and smaller.

Eventually,

the learning rate becomes so tiny

that weight updates almost stop.

The optimizer has effectively stopped learning.

---

# Chapter 3 — Should We Really Remember Everything?

Let's think like optimizer designers.

Suppose a parameter behaved like this.

```text
Iterations 1–500

Very Large Gradients
```

Later,

```text
Iterations 501–100000

Very Small Gradients
```

Should the optimizer continue punishing this parameter forever,

simply because it had large gradients long ago?

Probably not.

Those early gradients have already influenced the parameter.

What matters more is

how the parameter is behaving **now**.

This leads to a new design question.

> **Should every historical gradient receive equal importance?**

---

# Chapter 4 — A Real-Life Analogy

Imagine you are a student in Grade 10.

Which previous classes help you the most?

Certainly,

Grades 8 and 9.

Does Grade 1 matter?

Yes.

But only indirectly.

Everything you learned in Grade 1

has already been incorporated into later classes.

Recent learning is much more useful than distant learning.

Optimization behaves similarly.

Recent gradients often provide more useful information

than gradients computed thousands of iterations ago.

---

# Chapter 5 — Designing a Better Memory

Now we know the properties we want.

A good optimizer should

- remember recent gradients,
- gradually forget older gradients,
- never discard history completely,
- require constant memory.

Notice something.

We cannot store every historical gradient.

For models with billions of parameters,

this would be computationally impossible.

Instead,

we need a smarter idea.

We keep only **one running memory**.

---

# Chapter 6 — First Attempt

Suppose

```text
Memory = New Gradient
```

This completely forgets history.

Too aggressive.

Another idea.

```text
Memory = Memory + New Gradient
```

This is exactly AdaGrad.

It never forgets.

Too much memory.

We need something in between.

---

# Chapter 7 — A Better Idea

Instead of treating old and new information equally,

suppose we write

```text
New Memory

=

0.9 × Old Memory

+

0.1 × New Gradient
```

Something beautiful happens.

The previous memory is preserved,

but its influence becomes slightly weaker.

The newest gradient contributes immediately.

Every iteration,

older information gradually fades,

while recent information dominates.

Exactly the behaviour we wanted.

---

# Chapter 8 — Exponential Moving Average (EMA)

This idea is called an

> **Exponential Moving Average (EMA).**

More generally,

the equation becomes

```text
EMA = β × EMA + (1 - β) × g²
```

where

- **EMA** is the running estimate of squared gradients.
- **β** controls how quickly old information fades.
- **g²** is the squared gradient from the current iteration.

Notice the similarity with Momentum.

Both algorithms remember history.

The difference is

- Momentum remembers **velocity**.
- RMSProp remembers **squared gradients**.

---

# Chapter 9 — RMSProp Update Rule

Once the running average has been computed,

the effective learning rate becomes

```text
η_eff = η / √(EMA + ε)
```

where

```text
ε
```

is a tiny constant added for numerical stability,

preventing division by zero.

The weight update becomes

```text
W(new) = W(old) - η_eff × Gradient
```

or equivalently,

```text
W(new) = W(old) - [ η / √(EMA + ε) ] × Gradient
```

Unlike AdaGrad,

the denominator no longer grows forever.

Instead,

it reflects the **recent history of gradients**.

---

# Chapter 10 — Why RMSProp Works Better

AdaGrad remembers

every gradient equally.

RMSProp says

> "Recent gradients deserve more attention."

As training progresses,

old information naturally fades,

allowing the optimizer

to continue adapting throughout training.

This prevents the learning rate

from shrinking toward zero.

---

## Engineer's Insight

Notice how optimizer design has evolved.

```text
Gradient Descent
        │
        ▼
No memory
        │
        ▼
Momentum
        │
Remember previous movement
        │
        ▼
AdaGrad
        │
Remember all historical gradients
        │
        ▼
RMSProp
        │
Remember recent gradient history
```

Every optimizer exists because the previous optimizer had a limitation.

---

# Key Takeaways

- AdaGrad's accumulated gradient history grows forever.
- This causes the effective learning rate to continually decrease.
- RMSProp solves this by remembering recent gradients more than older ones.
- Exponential Moving Average provides a mathematically elegant way to implement fading memory.
- RMSProp adapts throughout training instead of becoming increasingly conservative.