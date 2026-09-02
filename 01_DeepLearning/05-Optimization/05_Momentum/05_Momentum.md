# Momentum

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- Why was Momentum invented?
- What problem does Mini-Batch Gradient Descent still have?
- What is momentum in optimization?
- How does Momentum remember previous updates?
- Why does Momentum accelerate learning?
- Why does Momentum reduce oscillations?
- Why is Momentum inspired by physics?
- Why is Momentum used inside many modern optimizers?

---

# Chapter 1 — Mini-Batch Gradient Descent Still Has a Problem

In the previous lecture, we learned that Mini-Batch Gradient Descent combines the advantages of Batch Gradient Descent and Stochastic Gradient Descent.

It produces:

- Faster learning
- Smoother gradients
- Better GPU utilization

Yet training is still not perfect.

Imagine descending a mountain.

At every step, the optimizer asks only one question:

```text
What is the gradient here?
```

Then it moves.

At the next step,

it asks exactly the same question again.

```text
Gradient

↓

Move

↓

Gradient

↓

Move

↓

Gradient

↓

Move
```

Notice something.

The optimizer has **no memory**.

Every update depends only on the current gradient.

It completely forgets every previous step.

---

# Chapter 2 — A Real-Life Analogy

Imagine driving a car.

Every second,

you completely forget

- your current speed
- the direction you were moving
- whether you were accelerating or slowing down

Instead,

every second,

you only look one meter ahead.

What would happen?

```text
←

→

←

→

←

→
```

You would constantly overcorrect.

The ride would become unstable.

This is exactly what happens during optimization.

The optimizer keeps reacting to the current gradient without remembering where it has already been moving.

---

# Chapter 3 — Engineers Asked a Brilliant Question

Instead of using only

```text
Current Gradient
```

what if we also remembered

```text
Previous Direction
```

or even

```text
Previous Speed
```

Instead of forgetting everything,

the optimizer could build up momentum.

This idea comes directly from physics.

---

# Chapter 4 — Understanding Momentum Using Physics

Imagine pushing a heavy shopping cart.

The first push

barely moves it.

```
Push

↓

Small Movement
```

Keep pushing in the same direction.

```
Push

↓

Push

↓

Push
```

Now the cart moves much faster.

Why?

Because it has built up **momentum**.

Now imagine changing direction suddenly.

The cart doesn't instantly turn around.

Its previous momentum resists the change.

Exactly the same idea is used in neural network optimization.

---

# Chapter 5 — How Momentum Works

Instead of updating weights using only

```text
Current Gradient
```

Momentum combines

```text
Previous Update

+

Current Gradient
```

The optimizer now remembers where it has been moving.

Conceptually,

the process becomes

```text
Previous Velocity

+

Current Gradient

↓

New Velocity

↓

Update Weights
```

Instead of making completely independent decisions,

every update is influenced by previous updates.

---

# Chapter 6 — Why Momentum Accelerates Learning

Suppose every gradient points roughly in the same direction.

Without Momentum

```text
→

→

→

→
```

Every step has the same size.

With Momentum

```text
→

→→

→→→

→→→→
```

Each update builds upon the previous one.

The optimizer accelerates in directions that consistently reduce the loss.

Training becomes faster.

---

# Chapter 7 — Why Momentum Reduces Oscillations

Now imagine a narrow valley.

The gradients look like

```text
↗

↘

↗

↘

↗

↘
```

Without Momentum,

the optimizer keeps bouncing from one side to the other.

With Momentum,

the previous velocity smooths these oscillations.

Instead of reacting to every tiny change,

the optimizer maintains a more consistent direction.

The optimization path becomes much smoother.

---

# Chapter 8 — The Mathematical Formula

Momentum introduces a new quantity called **velocity**.

Instead of directly updating the weights,

we first update the velocity.

```text
v = βv − η∇L
```

Then update the weights.

```text
w = w + v
```

Where:

```text
v  = Velocity

β  = Momentum coefficient

η  = Learning Rate

∇L = Gradient
```

Do not memorize this formula.

The important idea is:

> Velocity remembers previous updates.

The formula simply implements that intuition.

---

# Chapter 9 — Choosing the Momentum Coefficient

The momentum coefficient is usually written as

```text
β
```

Common values are

```text
0.9

0.95

0.99
```

A larger value means

- more memory
- smoother optimization
- stronger momentum

A smaller value means

- less memory
- reacts faster to new gradients

In practice,

```text
β = 0.9
```

works well for many deep learning problems.

---

# Chapter 10 — Why Modern Optimizers Use Momentum

Momentum was one of the first major improvements to Gradient Descent.

Today,

many famous optimizers build upon this idea.

Examples include:

- Momentum SGD
- Nesterov Momentum
- RMSProp
- Adam
- AdamW

Even though these optimizers become more sophisticated,

they all retain the central idea:

> **Remember previous updates instead of treating every gradient independently.**

---

# Engineering Insight — Understanding the Momentum Equation

Many students memorize the Momentum equation without truly understanding what each term represents.

```text
v = βv − η∇L
```

Let's build the intuition from first principles.

---

## What does the second `v` represent?

The `v` on the right-hand side is **not** a new velocity.

It is the velocity computed during the **previous optimization step**.

Think of it as:

```text
Previous Velocity

+

Current Gradient Update

↓

New Velocity
```

The optimizer remembers how it was already moving.

Instead of starting from scratch at every iteration, it continues moving in a similar direction.

---

## Why don't we simply add the previous velocity?

Suppose the equation were

```text
v = v − η∇L
```

Every iteration, the optimizer would keep accumulating velocity forever.

For example,

```text
Iteration 1

Velocity = 2

↓

Iteration 2

Velocity = 4

↓

Iteration 3

Velocity = 6

↓

Iteration 4

Velocity = 8
```

The optimizer would continue accelerating indefinitely and would likely overshoot the minimum.

We therefore introduce a decay factor.

```text
v = βv − η∇L
```

where

```text
0 < β < 1
```

Only a fraction of the previous velocity is retained.

---

## What does β actually mean?

The momentum coefficient β controls how much of the previous movement should be remembered.

For example,

```text
β = 0
```

means

```text
Remember Nothing
```

The equation becomes

```text
v = −η∇L
```

which reduces to ordinary Gradient Descent.

---

If

```text
β = 1
```

the optimizer remembers **everything**.

No previous movement is ever forgotten.

The velocity can continue growing indefinitely, making optimization unstable and causing the optimizer to overshoot the minimum.

---

A value such as

```text
β = 0.9
```

means

```text
Remember 90% of Yesterday

Forget 10%

Listen to Today's Gradient
```

This creates the perfect balance between:

- remembering useful previous movement
- adapting to new information

---

## Momentum is a Feedback Loop

One subtle but important observation is that the current gradient is itself influenced by the previous update.

Consider two consecutive iterations.

### Iteration 1

```text
Weights (W₁)

↓

Forward Pass

↓

Loss

↓

Gradient (∇L₁)

↓

Velocity (v₁)

↓

Updated Weights (W₂)
```

---

### Iteration 2

The optimizer no longer computes the gradient using `W₁`.

Instead,

```text
Weights (W₂)

↓

Forward Pass

↓

New Loss

↓

New Gradient (∇L₂)

↓

Velocity (v₂)
```

Therefore,

the previous velocity influences the new weights,

the new weights influence the new gradient,

and the new gradient influences the next velocity.

The optimization process forms a continuous feedback loop.

```text
Weights

↓

Predictions

↓

Loss

↓

Gradient

↓

Velocity

↓

Updated Weights

↓

Repeat...
```

Understanding this loop is much more valuable than memorizing the Momentum equation.

It explains how optimization evolves over thousands or millions of iterations.

---

# Research Insight

Momentum introduces one of the most fundamental ideas in machine learning:

```text
Remember the Past

vs

Adapt to the Present
```

This same design principle appears throughout AI.

Examples include:

- Momentum Optimizer
- RMSProp
- Adam
- AdamW
- RNN Hidden State
- Transformer KV Cache
- Reinforcement Learning Agents
- Memory-Augmented AI Systems

Once you recognize this pattern, you'll begin to notice it throughout modern AI architectures.

---

# Engineer Q&A

## Q1. Imagine you are one of the researchers before Momentum was invented.

Mini-Batch Gradient Descent is already working.

However, after every iteration the optimizer behaves like this:

```text
Gradient

↓

Update

↓

Forget Everything

↓

Gradient

↓

Update

↓

Forget Everything
```

### Question

Why is this an inefficient way to optimize a neural network?

What simple idea would you propose to improve it?

### Answer

Mini-Batch Gradient Descent updates the weights using only the current gradient.

After every iteration it completely forgets how it was moving previously.

Essentially, the optimizer asks the same two questions every iteration:

- Which direction should I move?
- How large should my step be?

but it never uses its previous movement to guide future updates.

This is inefficient because neural networks themselves learn by correcting previous mistakes.

If the model learns from history, why shouldn't the optimizer also learn from its previous movement?

Momentum solves exactly this problem by remembering part of the previous movement and combining it with the current gradient.

---

## Q2. Explain Momentum using a real-life example.

### Answer

Momentum comes directly from physics.

Consider pushing a heavy shopping cart.

Initially it is difficult to move because of inertia.

After several pushes,

the cart gains momentum and becomes much easier to keep moving.

The same idea applies to optimization.

As gradients consistently point in the same direction,

Momentum accumulates this movement,

allowing the optimizer to accelerate toward the minimum.

Another analogy is cricket.

A batsman usually spends the first few deliveries understanding the pitch, bounce and pace.

Once he gets settled,

he starts batting much more freely.

His previous adjustments help him play future deliveries better.

Similarly,

Momentum allows the optimizer to benefit from previous movement instead of starting from scratch every iteration.

---

## Q3. Suppose the gradients over several iterations are

```text
→

→

→

→

→
```

How will Gradient Descent and Momentum behave?

### Answer

Ordinary Gradient Descent takes approximately the same size step every iteration.

```text
→

→

→

→
```

Momentum builds upon the previous movement.

```text
→

→→

→→→

→→→→
```

The optimizer accelerates whenever gradients consistently point in the same direction.

This leads to faster convergence.

---

## Q4. Suppose the gradients look like

```text
↗

↘

↗

↘

↗

↘
```

Why does Momentum produce smoother optimization?

### Answer

Ordinary Gradient Descent reacts only to the current gradient.

As a result,

the optimizer continuously changes direction,

causing oscillations.

Momentum remembers part of the previous movement.

Instead of reacting aggressively to every gradient change,

it combines previous movement with the current gradient,

making the optimization path much smoother.

The primary reason for these oscillations is often the geometry of the loss surface,

especially narrow valleys.

Mini-batch noise can make these oscillations even more pronounced.

---

## Q5. Why introduce velocity?

Momentum uses

```text
v = βv − η∇L
```

instead of directly updating the weights.

Why?

### Answer

Velocity stores information about previous movement.

The optimizer no longer depends only on the current gradient.

Instead,

it combines

```text
Previous Movement

+

Current Gradient
```

This allows optimization to become faster,

more stable,

and less sensitive to sudden gradient changes.

---

## Q6. What happens if

```text
β = 0
```

### Answer

The previous velocity is completely discarded.

The Momentum equation becomes

```text
v = −η∇L
```

which reduces to ordinary Gradient Descent.

This shows that β controls how much information from previous movement should be preserved.

---

## Q7. What happens if

```text
β = 1
```

### Answer

The optimizer remembers 100% of the previous movement.

This is generally undesirable.

If a previous update was noisy,

the optimizer carries all of that movement into the next iteration.

Velocity can continue growing,

making optimization unstable and increasing the chance of overshooting the minimum.

A value around

```text
β = 0.9
```

provides a better balance between remembering useful history and adapting to new gradients.

---

## Q8. Explain Momentum without using the equation.

Use only the sentence:

```text
New Velocity

=

Most of Yesterday's Movement

+

Today's Gradient
```

### Answer

Momentum combines previous movement with the current gradient.

Instead of making every decision independently,

the optimizer remembers where it has been moving,

while still allowing the current gradient to influence future updates.

This creates smoother,

more stable,

and faster optimization.

---

## Q9. Why is optimization a feedback loop?

Consider

```text
Weights

↓

Predictions

↓

Loss

↓

Gradient

↓

Velocity

↓

Updated Weights

↓

Repeat
```

### Answer

Every weight update changes the model's predictions.

New predictions change the loss.

The new loss produces a new gradient.

The new gradient combines with the previous velocity,

creating another weight update.

Therefore,

each iteration influences every future iteration.

Optimization is therefore a continuous feedback loop rather than a sequence of independent updates.

---

# Key Takeaways

- Mini-Batch Gradient Descent still reacts only to the current gradient.
- Momentum introduces memory into optimization.
- Previous updates influence future updates.
- Momentum accelerates movement in consistent directions.
- Momentum reduces oscillations in narrow valleys.
- Most modern optimizers build upon the idea of Momentum.