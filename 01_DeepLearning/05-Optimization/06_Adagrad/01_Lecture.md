# 06. AdaGrad (Adaptive Gradient)

---

# The Problem Momentum Could Not Solve

Momentum was a huge improvement over Gradient Descent.

It remembered previous movement.

It accelerated optimization.

It reduced oscillations.

Training became much smoother.

Problem solved?

**Not quite.**

Researchers soon discovered another limitation.

---

# Imagine Training a Neural Network

Suppose our neural network has only two weights.

```text
W1

W2
```

During training we observe something interesting.

---

## Weight 1

Every iteration receives a very large gradient.

```text
Iteration 1

Gradient = 120

Iteration 2

Gradient = 95

Iteration 3

Gradient = 140
```

---

## Weight 2

Hardly changes.

```text
Iteration 1

Gradient = 0.001

Iteration 2

Gradient = 0.003

Iteration 3

Gradient = 0.002
```

Yet our optimizer says

```text
Learning Rate

=

0.001
```

for both weights.

---

# Engineering Question

Does this make engineering sense?

Should both parameters learn at exactly the same speed?

Think about it.

One parameter is already changing rapidly.

The other is barely moving.

Why should they receive the same learning rate?

---

# Real-Life Analogy

Imagine two cars.

Car A

```text
Speed = 180 km/hr
```

Car B

```text
Speed = 15 km/hr
```

Would you press the accelerator equally hard on both?

Of course not.

The fast-moving car requires smaller adjustments.

The slow-moving car needs larger encouragement.

Optimization is no different.

---

# The Hidden Assumption

Every optimizer we have studied so far assumes

```text
One Learning Rate

↓

For Every Parameter
```

This assumption works reasonably well.

But is it the best possible strategy?

Researchers started asking a different question.

> **Should every parameter have its own learning rate?**

This was a revolutionary shift in thinking.

---

# A Better Question

Instead of asking

```text
What is the best learning rate?
```

they asked

```text
How can each parameter decide its own learning rate?
```

Notice the difference.

We're no longer searching for

```text
η = 0.001
```

We're searching for

```text
η₁

η₂

η₃

...

ηₙ
```

Every parameter can learn differently.

---

# How Should We Decide?

Suppose a parameter *repeatedly* receives large gradients.

```text
120

110

95

130

140
```

What does this tell us?

It tells us

> This parameter is already changing aggressively.

Should we continue taking large steps?

Probably not.

We should become more careful.

---

Now consider another parameter.

```text
0.001

0.002

0.001

0.003
```

This parameter barely changes.

Should we reduce its learning rate?

Probably not.

Perhaps we should allow it to learn faster.

---

# Engineering Requirement

We now have a clear requirement.

If a parameter *repeatedly* receives

```text
Large Gradients
```

↓

Its learning rate should decrease.

---

If a parameter *repeatedly* receives

```text
Small Gradients
```

↓

Its learning rate should remain larger.

---

Notice something important.

We are no longer making decisions based only on the **current gradient**.

We are looking at the **history of gradients**.

That is the key insight behind AdaGrad.

---

# What Information Should We Remember?

Momentum remembers

```text
Previous Velocity
```

AdaGrad asks a different question.

What if we remember

```text
Previous Gradients
```

instead?

Not to determine the direction.

But to determine the **learning rate**.

Very clever.

---

# The Core Philosophy

Momentum remembers movement.

AdaGrad remembers gradient history.

Momentum changes

```text
Direction of Movement
```

AdaGrad changes

```text
Step Size
```

These optimizers solve two completely different engineering problems.

---

# The Big Idea

Instead of

```text
One Learning Rate

↓

All Parameters
```

AdaGrad proposes

```text
Every Parameter

↓

Own Learning Rate
```

Parameters that have already moved a lot should slow down.

Parameters that rarely move should continue learning.

---

# Why Is It Called Adaptive Gradient?

Because the learning rate automatically **adapts**

based on the history of gradients.

No longer fixed.

No longer identical for every parameter.

The optimizer continuously adjusts the learning rate during training.

Hence the name

```text
Adaptive Gradient

↓

AdaGrad
```

---

# Engineering Insight

Notice how optimization algorithms are evolving.

Gradient Descent asked

> Which direction should I move?

---

Momentum asked

> Can I use my previous movement to make better decisions?

---

AdaGrad asks

> Should every parameter learn at the same speed?

Every new optimizer is born because engineers discovered a limitation in the previous one.

# Mathematical Derivation

Until now, we have only discussed the intuition behind AdaGrad.

Now, let's derive the mathematics instead of memorizing the final equation.

---

## Engineering Requirements

From our discussion, we established the following requirements.

### Requirement 1

If a parameter repeatedly receives **large gradients**,

its learning rate should gradually **decrease**.

---

### Requirement 2

If a parameter usually receives **small gradients**,

its learning rate should remain **relatively larger**.

---

### Requirement 3

The learning rate should depend on the **history of gradients**, not just the current gradient.

---

The next question becomes:

> **How should we represent the history of gradients?**

---

# Candidate 1 — Sum of Gradients

One simple idea is to store

```text
g₁ + g₂ + g₃ + ...
```

where `g` represents the gradient.

### Problem

Suppose the gradients are

```text
10
-10
10
-10
```

The sum becomes

```text
0
```

Does this mean the parameter never received large gradients?

No.

The positive and negative gradients simply cancel each other.

This is exactly the same **error cancellation problem** we encountered while studying Mean Error.

Therefore,

**Sum of gradients is not a reliable representation of gradient history.**

---

# Candidate 2 — Average Gradient

Another idea is to calculate

```text
(g₁ + g₂ + g₃ + ...)/N
```

This gives the average historical gradient.

### Problem

The numerator is still the sum of gradients.

Therefore,

positive and negative gradients continue to cancel each other.

The averaging operation does not solve the fundamental problem.

---

# Candidate 3 — Absolute Gradients

To avoid error cancellation, we can store

```text
|g₁|

+

|g₂|

+

|g₃|
```

or even use the Mean Absolute Gradient.

This removes the cancellation issue and preserves the magnitude of gradients.

However,

absolute values treat all gradients linearly.

A gradient of 100 is considered only ten times larger than a gradient of 10.

Should very large gradients receive much stronger importance?

Maybe.

Let's explore another option.

---

# Candidate 4 — Squared Gradients

Instead of absolute values,

suppose we store

```text
g₁²

+

g₂²

+

g₃²
```

Immediately we notice several advantages.

- Positive and negative gradients can no longer cancel each other.
- Large gradients become much larger.
- Small gradients remain relatively small.

For example,

```text
Gradient = 2

↓

Squared = 4
```

whereas

```text
Gradient = 20

↓

Squared = 400
```

Large gradients now contribute much more strongly to the accumulated history.

This is actually desirable.

If a parameter repeatedly experiences large gradients,

we want its future learning rate to decrease more aggressively.

---

# Cumulative History or Average History?

Now another engineering question appears.

Should we store

```text
Average Squared Gradient
```

or

```text
Cumulative Squared Gradient
```

Consider two parameters.

### Parameter A

Receives large gradients for only 10 iterations.

### Parameter B

Receives large gradients for 1000 iterations.

Should both receive the same learning rate?

Probably not.

A parameter that has been receiving large gradients for a long time has consistently demonstrated aggressive updates.

Its future learning rate should become smaller.

Therefore,

AdaGrad chooses to remember the **entire cumulative history**.

---

# Building the Accumulator

Let

```text
G
```

represent the accumulated gradient history.

Initially,

```text
G = 0
```

After every iteration,

we receive a new gradient.

Since we decided to store the cumulative squared gradients,

the update naturally becomes

```text
G = G + g²
```

Notice something important.

We did not memorize this equation.

We derived it from our engineering requirements.

---

# Designing the Adaptive Learning Rate

Now that we have accumulated gradient history,

how should it affect the learning rate?

Our requirement is

```text
Large History

↓

Smaller Learning Rate
```

This immediately suggests an inverse relationship.

One possible equation is

```text
η / G
```

Another possibility is

```text
η / √G
```

Which one should we choose?

---

# Why Not Divide by G?

Suppose

```text
η = 0.01

G = 10000
```

The learning rate becomes

```text
0.01 / 10000

=

0.000001
```

This is almost zero.

Learning practically stops.

The optimizer becomes far too conservative.

Remember,

`G` already contains squared gradients.

Large gradients have already been amplified.

Dividing by this enormous number punishes the parameter twice.

This slows learning far more than intended.

---

# Why Divide by √G?

Now consider

```text
√10000 = 100
```

The learning rate becomes

```text
0.01 / 100

=

0.0001
```

Learning still slows down,

but it does not stop.

The square root moderates the rapid growth caused by the squared gradients.

Instead of eliminating learning,

it simply makes learning more cautious.

This behavior perfectly matches our engineering requirements.

---

# Final AdaGrad Update

The accumulated gradient history is

```text
G = G + g²
```

The adaptive learning rate becomes

```text
η / √G
```

Finally,

the weight update equation becomes

```text
w = w - (η / √G) × g
```

where

- `g` is the current gradient.
- `G` is the cumulative sum of squared gradients.
- `η` is the initial learning rate.

Every parameter now automatically receives its own effective learning rate.

---

# Engineering Note — Where Does η = 0.01 Come From?

A common question is:

> **Why do most books start with η = 0.01?**

The answer is surprisingly simple.

There is no universal mathematical formula that always produces the perfect learning rate.

Originally,

the learning rate was treated as a **hyperparameter**.

Researchers experimented with values such as

```text
0.1

0.01

0.001

0.0001
```

and selected the value that produced stable training.

Through years of empirical experience,

`0.01` became a common starting point for many optimization algorithms.

However,

the ideal learning rate depends on many factors,

including

- model architecture,
- dataset,
- initialization,
- optimizer,
- batch size,
- activation functions,
- and the shape of the loss surface.

There is no single value that works for every neural network.

This limitation motivated the development of adaptive optimizers like AdaGrad.

Instead of forcing humans to manually choose the perfect learning rate,

AdaGrad automatically adjusts the effective learning rate for every parameter during training.


# Engineering Notes

## 1. AdaGrad Maintains a Separate Learning Rate for Every Parameter

Unlike traditional Gradient Descent,

which uses one global learning rate,

AdaGrad computes an effective learning rate independently for every parameter.

This allows frequently updated parameters to slow down,

while infrequently updated parameters continue learning faster.

---

## 2. Frequently Updated Parameters Learn More Cautiously

Whenever a parameter repeatedly receives large gradients,

its accumulated gradient history grows rapidly.

As a result,

its effective learning rate gradually decreases.

This prevents unstable updates and improves optimization stability.

---

## 3. Rare Features Continue Learning

Parameters that rarely receive gradients accumulate much smaller historical values.

Therefore,

their effective learning rate remains relatively large.

This makes AdaGrad especially useful for sparse datasets,

such as Natural Language Processing and recommendation systems,

where many features appear infrequently.

---

## 4. No Manual Per-Parameter Tuning

Without AdaGrad,

every parameter shares the same learning rate.

AdaGrad automatically adjusts the learning speed for each parameter,

eliminating the need for manually designing different learning rates.

---

## 5. Historical Knowledge Guides Future Learning

AdaGrad is one of the first optimizers that truly learns from training history.

Every optimization step depends not only on the current gradient,

but also on all previous gradients seen by that parameter.

---

## 6. Learning Rate Only Decreases

The accumulated squared gradient

```
G = G + g²
```

can only increase.

Since

```
Learning Rate = η / √G
```

the effective learning rate can only become smaller over time.

It never increases again.

---

## 7. Long Training Can Eventually Stop Learning

As training continues,

the accumulated gradient history becomes extremely large.

Eventually,

the effective learning rate becomes so small that parameter updates become almost negligible.

The optimizer may stop making meaningful progress,

even though training has not truly converged.

This is the biggest practical limitation of AdaGrad.

---

## 8. AdaGrad Inspired Future Optimizers

AdaGrad introduced one revolutionary idea:

> Learning rates should adapt automatically.

Although AdaGrad itself has limitations,

its central idea inspired RMSProp, Adam, and many modern optimization algorithms.

Nearly every adaptive optimizer used today traces its roots back to AdaGrad.

---

# Key Insight

Notice the progression of optimization algorithms.

Gradient Descent asked:

> Which direction should I move?

Momentum asked:

> Can I learn from my previous movement?

AdaGrad asked:

> Should every parameter learn at the same speed?

Every optimizer exists because engineers discovered a limitation in the previous one.

---

# Key Takeaways

- Momentum remembers previous movement.
- AdaGrad remembers previous gradients.
- Momentum accelerates optimization.
- AdaGrad adapts learning rates.
- Different parameters should not always learn at the same speed.
- Adaptive learning rates allow each parameter to learn according to its own optimization history.

---

# Think Like a Researcher

Instead of accepting a fixed learning rate, researchers asked:

> Can the optimizer decide how much each parameter should learn?

That single question led to AdaGrad.

This is an important engineering lesson.

Every breakthrough optimizer begins with a simple question about the limitations of the previous one.

---

# Coming Next

We now understand the intuition.

The next question is:

> **How should we mathematically use the history of gradients to calculate the learning rate?**

Should we store:

- Sum of gradients?
- Average of gradients?
- Absolute gradients?
- Squared gradients?
- Maximum gradient?

We'll derive the mathematics from first principles in the next section instead of memorizing the final equation.

# AdaGrad Algorithm

After understanding the intuition and deriving the mathematics, we can now summarize AdaGrad into a simple algorithm.

---

## Step 1: Initialize Parameters

Initialize the model weights.

Initialize the accumulated squared gradient history for every parameter.

```text
For every parameter:

G = 0
```

where:

- **G** stores the cumulative sum of squared gradients.
- Initially, no training has happened, so the history is zero.

---

## Step 2: Perform Forward Pass

Pass the current training batch through the neural network.

Compute the predictions.

---

## Step 3: Compute Loss

Calculate the loss using an appropriate loss function.

Examples include:

- Mean Squared Error (Regression)
- Cross Entropy Loss (Classification)

---

## Step 4: Compute Gradients

Using Backpropagation,

compute the gradient for every trainable parameter.

```text
g = ∂L / ∂w
```

This tells us

- the direction in which the loss increases most rapidly,
- and how sensitive the loss is to that parameter.

---

## Step 5: Update Gradient History

Instead of discarding previous gradients,

AdaGrad accumulates the squared gradients.

```text
G = G + g²
```

Notice that every parameter maintains its own independent history.

---

## Step 6: Compute Adaptive Learning Rate

For each parameter,

compute its effective learning rate.

```text
η_eff = η / √G
```

where:

- **η** is the initial learning rate.
- **η_eff** is the adaptive learning rate used for this parameter.

Parameters with large historical gradients automatically receive smaller learning rates.

Parameters with small historical gradients continue learning faster.

---

## Step 7: Update the Weights

Finally,

update the weights using the adaptive learning rate.

```text
w = w − η_eff × g
```

Unlike ordinary Gradient Descent,

every parameter now moves with its own learning rate.

---

# Complete AdaGrad Update Rule

For every training iteration:

```text
Compute gradient:

g = ∂L / ∂w

↓

Update accumulated history:

G = G + g²

↓

Compute adaptive learning rate:

η_eff = η / √G

↓

Update weights:

w = w − η_eff × g
```

---

# Why AdaGrad Works

AdaGrad automatically adjusts the learning rate for every parameter based on its training history.

Parameters that have received large gradients repeatedly become more cautious over time.

Parameters that rarely receive updates continue learning aggressively.

As a result,

AdaGrad eliminates the need for manually assigning different learning rates to different parameters.

# Optimizer Summary

## Design Goal

Adapt the learning rate independently for every parameter.

---

## Core Idea

Parameters that have learned a lot should slow down, while parameters that have learned very little should continue learning faster.

---

## Mathematical Innovation

Maintain a cumulative sum of squared gradients and divide the learning rate by the square root of this accumulated history.

---

## Strengths

- Automatic per-parameter learning rates.
- Excellent for sparse datasets.
- Reduces aggressive updates for frequently changing parameters.
- Requires less manual tuning than vanilla Gradient Descent.

---

## Limitations

- Gradient history only grows.
- Learning rate continuously decreases.
- Training may eventually become extremely slow.
- Not suitable for very long training runs.

---

## Best Use Cases

- Sparse feature spaces.
- NLP applications.
- Recommendation systems.
- Problems where some parameters are updated much less frequently than others.

---

## When NOT to Use

- Very long training schedules.
- Large-scale LLM pretraining.
- Situations where continuous adaptation is required throughout training.

---

## Interview One-Liner

AdaGrad is an adaptive optimization algorithm that assigns each parameter its own learning rate by accumulating squared gradients, allowing frequently updated parameters to slow down while rare parameters continue learning efficiently.

---

# Key Takeaways

- AdaGrad remembers the **entire history** of squared gradients.
- Every parameter maintains its **own adaptive learning rate**.
- Frequently updated parameters automatically slow down.
- Rarely updated parameters continue learning efficiently.
- This makes AdaGrad especially useful for sparse datasets.
- The main limitation is that the accumulated history **never decreases**, causing the learning rate to become extremely small during long training runs.