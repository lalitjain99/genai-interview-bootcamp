# Learning Rate — How Big Should We Move?

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- What is the Learning Rate?
- Why do we need a Learning Rate?
- What happens if it is too small?
- What happens if it is too large?
- Why can't we always choose a large Learning Rate?
- Why can't we always choose a tiny Learning Rate?
- How is the Learning Rate different from the Gradient?
- Why do modern optimizers adapt the Learning Rate automatically?

---

# Chapter 1 — We Already Know the Direction

In the previous module, we learned Gradient Descent.

The weight update equation is

$$
w = w - \eta \nabla L
$$

At first glance,

this equation contains two important quantities:

```text
Gradient (∇L)

Learning Rate (η)
```

We already understand the Gradient.

It tells us

> **Which direction increases the loss the fastest.**

Therefore,

moving in the opposite direction decreases the loss.

So the Gradient answers one question:

```text
Which direction should we move?
```

But another important question still remains.

```text
How far should we move?
```

That is exactly what the Learning Rate decides.

---

# Chapter 2 — An Analogy

Imagine you're standing on top of a mountain.

Your goal is to reach the lowest point in the valley.

Fortunately,

someone tells you the correct direction.

```text
↓

↓

↓

Valley
```

Great!

Now you know **where** to go.

But another question immediately appears.

```text
Should I take

tiny steps?

or

giant jumps?
```

Knowing the direction alone is not enough.

You also need to decide the size of each step.

This is exactly the purpose of the Learning Rate.

---

# Chapter 3 — Separating Two Different Ideas

Many beginners confuse the Gradient with the Learning Rate.

They are completely different.

Think of hiking down the mountain.

The Gradient tells you:

```text
Walk south-west.
```

The Learning Rate tells you:

```text
Take steps of 1 meter.
```

One decides **direction**.

The other decides **step size**.

Together they produce movement.

---

# Chapter 4 — What Happens If the Learning Rate Is Too Small?

Suppose

```text
η = 0.000001
```

Every weight update becomes extremely tiny.

```text
Current Weight

↓

Tiny Update

↓

Almost Same Weight
```

The model is moving in the correct direction,

but so slowly

that training may take weeks,

months,

or even never converge within a practical amount of time.

Imagine trying to walk one kilometer

by moving only

```text
1 millimeter
```

at a time.

You will eventually reach the destination,

but it will take an enormous number of steps.

---

# Chapter 5 — What Happens If the Learning Rate Is Too Large?

Now suppose

```text
η = 100
```

Instead of careful steps,

the optimizer makes giant jumps.

Imagine trying to walk down a mountain,

but every step is

```text
100 meters long.
```

What happens?

You repeatedly jump over the valley.

```text
      Valley

        ▼

----->     <-----
```

Instead of getting closer,

you keep overshooting.

Sometimes,

the loss may even become larger after every update.

Training becomes unstable.

This phenomenon is called

> **Overshooting the Minimum.**

---

# Chapter 6 — The Sweet Spot

A good Learning Rate finds a balance.

```text
Too Small

↓

Very Slow Learning

-------------------------

Just Right

↓

Fast and Stable Learning

-------------------------

Too Large

↓

Unstable Training
```

The ideal Learning Rate allows the optimizer to

- move quickly when appropriate,
- remain stable,
- and eventually converge to a minimum.

---

# Chapter 7 — But Didn't We Already Say the Gradient Gets Smaller?

Earlier,

we learned an important property of MSE.

As the prediction improves,

the Gradient naturally becomes smaller.

```text
Large Error

↓

Large Gradient

↓

Large Update
```

Later,

```text
Small Error

↓

Small Gradient

↓

Small Update
```

A natural question arises.

> **If the Gradient already controls the update size, why do we need a Learning Rate at all?**

This is an excellent question.

The Gradient tells us

how sensitive the loss is to a particular weight.

But we may not want to trust it completely.

Think of driving a car.

Your GPS tells you

"Turn right."

That is the direction.

But the GPS does **not** decide

whether you should drive at

```text
20 km/h

or

120 km/h
```

Speed is a separate decision.

Similarly,

the Gradient provides a suggested update.

The Learning Rate determines

how aggressively we follow that suggestion.

---

# Chapter 8 — Another Way to Think About It

Suppose the Gradient is

```text
50
```

Without a Learning Rate,

the weight update becomes

```text
Δw = -50
```

That may be far too large.

Instead,

choose

```text
η = 0.01
```

Now,

```text
Δw = -0.5
```

Much safer.

The Learning Rate acts like a scaling factor.

It controls the magnitude of every update,

without changing the direction.

---

# Chapter 9 — Why Not Change the Learning Rate Automatically?

So far,

we've assumed that the Learning Rate is fixed.

But think about our mountain analogy.

When we're far from the valley,

large steps are helpful.

Near the valley,

smaller steps are safer.

Wouldn't it be nice if the optimizer automatically adjusted its step size?

That question led researchers to develop more advanced optimization algorithms.

In the upcoming lectures,

we'll discover how:

- Momentum
- AdaGrad
- RMSProp
- Adam

all improve upon basic Gradient Descent,

largely by changing how the Learning Rate behaves.

---

# Interview Insight

A common interview question is:

> **If the Gradient already determines the update, why do we still need a Learning Rate?**

A strong answer is:

The Gradient tells us the direction and sensitivity of the loss with respect to the weights.

However,

blindly following the Gradient may result in updates that are too large or too small.

The Learning Rate scales the Gradient,

allowing us to control how aggressively the model updates its parameters while maintaining stable learning.

---

# Key Takeaways

- The Gradient tells us **which direction** to move.
- The Learning Rate tells us **how far** to move.
- A very small Learning Rate leads to slow convergence.
- A very large Learning Rate causes unstable training and overshooting.
- The Learning Rate scales the Gradient without changing its direction.
- Choosing an appropriate Learning Rate is one of the most important decisions in training a neural network.
- Modern optimizers improve learning by adapting the Learning Rate automatically.


## Engineer's Insight

A common misconception is that Gradient Descent simply follows the gradient.

In reality, it does **not** blindly follow it.

The Gradient tells us the **direction** and the **natural magnitude** suggested by the loss surface.

However, blindly trusting this update may lead to unstable learning.

The Learning Rate acts as a safety controller.

It asks:

> "The Gradient suggests moving this much.
> Should we really trust it completely?"

Instead of accepting the Gradient's update directly, we scale it using the Learning Rate.

This separation between **direction** and **step size** is one of the key ideas behind modern optimization algorithms.


---

# 🧠 Engineer Discussion & Intuitive Q&A

## Q1 — Does a Smaller Learning Rate Always Mean Better Learning?

Suppose the Gradient is

```text
∇L = 50
```

Two identical neural networks are trained.

**Network A**

```text
Learning Rate = 0.1
```

**Network B**

```text
Learning Rate = 0.0001
```

Which network will reach the minimum faster?

Will both eventually converge?

### Answer

The Gradient tells us **which direction** to move, while the Learning Rate determines **how large each update should be**.

The weight update equation is

```text
w(new) = w(old) - η × ∇L
```

For Network A,

```text
Update = 0.1 × 50 = 5
```

For Network B,

```text
Update = 0.0001 × 50 = 0.005
```

Network B makes extremely small updates.

Its weights barely change after each iteration.

Although it is moving in the correct direction, it may require an enormous number of iterations before converging.

In practice, training may become computationally infeasible long before convergence is reached.

---

## Q2 — Can We Always Increase the Learning Rate if the Gradient is Small?

Suppose during one iteration,

```text
Gradient = 0.0001

Learning Rate = 1000
```

The update becomes

```text
0.0001 × 1000 = 0.1
```

which seems perfectly reasonable.

Does this mean increasing the Learning Rate always solves slow learning?

### Answer

No.

The Gradient is **not constant throughout training**.

It changes after every weight update.

A large Learning Rate that works well for a tiny Gradient today may become extremely dangerous if the Gradient becomes much larger in later iterations.

For example,

```text
Gradient = 20

Learning Rate = 1000
```

produces

```text
Update = 20,000
```

Such a huge update may completely overshoot the minimum and cause training to become unstable.

Learning Rate must therefore be chosen considering the entire optimization process, not just a single iteration.

---

## Q3 — GPS Analogy

Imagine driving to an unfamiliar destination.

Your GPS tells you

```text
Turn Right
```

while your car is travelling at

```text
40 km/h
```

Which part represents the Gradient?

Which part represents the Learning Rate?

### Answer

The GPS represents the **Gradient**.

It tells us the correct direction to move.

The speed of the car represents the **Learning Rate**.

It determines how aggressively we follow that direction.

Both are equally important.

Knowing the direction without controlling speed leads to inefficient or unsafe travel.

Similarly,

speed without knowing the direction is equally useless.

Successful optimization requires both.

---

## Q4 — Why Do We Need a Learning Rate at All?

Suppose we remove the Learning Rate entirely.

The update rule becomes

```text
w(new) = w(old) - ∇L
```

Would Gradient Descent still work?

### Answer

Yes.

Gradient Descent would still update the weights.

However,

the Gradient would now be responsible for **two completely different decisions**:

- Which direction to move.
- How large the update should be.

These are independent decisions.

The Learning Rate separates them.

Instead of blindly trusting the Gradient's suggested update,

we introduce a scaling factor that controls how aggressively we move.

This separation makes optimization significantly more stable and controllable.

---

## Q5 — Why Can't We Simply Use a Very Large Learning Rate?

Suppose someone argues,

> "Larger steps reach the destination faster, so let's always use a Learning Rate of 100."

How would you respond?

### Answer

A very large Learning Rate ignores the geometry of the loss surface.

When the optimizer is far from the minimum,

large updates may be acceptable.

However,

as it approaches the minimum,

the optimizer must make increasingly careful adjustments.

If the Learning Rate remains excessively large,

the optimizer repeatedly jumps across the minimum instead of settling into it.

This phenomenon is known as **overshooting the minimum** and often causes unstable training.

---

## 🎯 Staff Engineer Challenge

You are training a **70B parameter Large Language Model**.

The training loss behaves as follows:

```text
10.2

↓

8.7

↓

7.9

↓

7.1

↓

15.3

↓

28.6

↓

NaN
```

Training was progressing normally for several hours before suddenly diverging.

The team immediately suggests reducing the Learning Rate.

Would that be your first action?

### Discussion

A Staff Engineer begins by returning to first principles.

Weight updates are determined by

```text
Weight Update = Learning Rate × Gradient
```

Therefore,

an unstable update can occur because

- the Learning Rate is too large,
- the Gradient has become unstable,
- or both together produce excessively large updates.

Rather than immediately reducing the Learning Rate,

a systematic investigation should begin.

Questions to ask include:

- Has the Learning Rate changed because of a scheduler?
- Have the Gradient values suddenly exploded?
- Did a problematic training batch appear?
- Has numerical instability occurred?
- Is the optimizer behaving unexpectedly?

Only after identifying the underlying cause should the Learning Rate be adjusted.

---

## 💡 Engineer's Insight

A common misconception is that Gradient Descent simply follows the Gradient.

It does **not** blindly trust it.

The Gradient tells us the **direction** and the **natural magnitude** suggested by the loss surface.

The Learning Rate acts as a safety controller.

It asks:

> "The Gradient suggests moving this much.

> Should we really trust it completely?"

Instead of accepting the Gradient's update directly,

we scale it using the Learning Rate.

This separation between **direction** and **step size** is one of the fundamental ideas behind modern optimization algorithms.