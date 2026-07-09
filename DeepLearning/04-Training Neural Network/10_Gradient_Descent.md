# Gradient Descent — How Does the Model Learn from Its Mistakes?

## Module 04 — Training Neural Networks

## Learning Objectives

By the end of this lecture, you should be able to answer:

- The loss function tells us the model is wrong—but how does the model fix itself?
- What is Gradient Descent?
- What is an optimization algorithm?
- Why is it called "Gradient Descent"?
- What is the role of the learning rate?
- Why doesn't the model simply jump directly to the best weights?
- How does Gradient Descent fit into GPT training?

---

# Chapter 1 — The Report Card Problem

In the previous lecture, we learned how to calculate the loss.

Imagine the model predicts:

```text
House Price = ₹80 Lakhs
```

Actual price:

```text
₹1 Crore
```

The loss function computes

```text
Loss = 400
```

Great.

Now the model knows:

> "I made a mistake."

But immediately another question appears.

> **How should the model fix that mistake?**

The loss function only gives us a number.

It doesn't tell us

- which weight is wrong,
- how much it is wrong,
- or whether the weight should increase or decrease.

So who decides that?

That is the job of **Gradient Descent**.

---

# Chapter 2 — Imagine You're Climbing a Mountain

Imagine you are standing on top of a mountain.

Unfortunately...

Your goal is not to reach the top.

Your goal is to reach the **lowest point in the valley**.

But there is one problem.

Heavy fog covers the mountain.

You cannot see the entire landscape.

You can only observe the ground immediately around your feet.

So how do you reach the valley?

Very simple.

Take one small step in the direction where the ground slopes downward the most.

Then repeat.

Again.

Again.

Again.

Eventually...

You reach the bottom.

---

# Chapter 3 — The Mountain Is Actually the Loss Function

Now replace the mountain with something else.

Instead of height,

imagine the vertical axis represents

```text
Loss
```

```text
          High Loss

              ▲

             / \

            /   \

           /     \

          /       \

         /         \

        ▼

     Lowest Loss
```

Every point on this mountain represents a different set of model weights.

Some weights produce huge loss.

Some produce very small loss.

Our goal is simple.

Find the weights that produce the **smallest possible loss**.

---

# Chapter 4 — Why Is It Called Gradient Descent?

The name contains two words.

## Gradient

The **gradient** tells us the direction of the steepest increase.

Imagine placing a ball on a hill.

The gradient points uphill.

If we walk in the opposite direction,

we move downhill.

---

## Descent

Descent simply means

```text
Going Down
```

Putting both together,

Gradient Descent means

> **Move in the direction that reduces the loss the fastest.**

# Before We Define the Gradient

Imagine you're standing on a foggy mountain.

You want to reach the valley, but you can't see the entire mountain.

You can only observe the slope immediately around your feet.

Question:

> If you wanted to climb UP as quickly as possible, which direction would you choose?

Naturally, you'd choose the direction where the ground rises the fastest.

Mathematicians call this direction the **gradient**.

But our goal is not to climb.

Our goal is to reach the valley.

So instead of moving **with** the gradient, we move **against** it.

This simple idea is called **Gradient Descent**.

---

# Chapter 5 — One Small Step at a Time

A natural question is:

> "If the model knows where the valley is, why not jump directly there?"

Because it doesn't.

The model can only estimate the slope **at its current position**.

So it updates the weights little by little.

```text
Current Weights

↓

Calculate Loss

↓

Find Downhill Direction

↓

Take Small Step

↓

New Weights

↓

Repeat
```

Learning is therefore an **iterative process**.

The model improves gradually.

Not instantly.

---

# Chapter 6 — The Learning Rate

Suppose you are walking down a mountain.

Should every step be

```text
1 meter?
```

or

```text
100 meters?
```

Large steps are dangerous.

You might jump over the valley.

Tiny steps are safe.

But learning becomes painfully slow.

This introduces one of the most important hyperparameters in Deep Learning.

The **Learning Rate**.

The learning rate controls

> **How big should each weight update be?**

---

## Learning Rate Too Small

```text
🙂

Step...

Step...

Step...

Step...
```

Very safe.

Very slow.

---

## Learning Rate Too Large

```text
↘

↗

↘

↗

↘
```

The model keeps jumping from one side of the valley to the other.

It may never converge.

---

## A Good Learning Rate

```text
↘

 ↘

  ↘

   ●
```

Steady progress.

Eventually reaches the minimum.

---

# Chapter 7 — Optimization Is Like Learning to Ride a Bicycle

Think about learning to ride a bicycle.

You don't become perfect on the first attempt.

You try.

You lose balance.

You adjust.

You try again.

Each attempt teaches you something.

Eventually,

your corrections become smaller and smaller until riding feels natural.

Gradient Descent works exactly the same way.

Every prediction produces feedback.

Every feedback slightly adjusts the weights.

Over millions of iterations,

the network gradually becomes better.

---

# Chapter 8 — The Complete Learning Cycle

We can now complete the story that began several lectures ago.

```text
Input

↓

Prediction

↓

Loss Function

↓

Loss

↓

Gradient Descent

↓

Update Weights

↓

Better Prediction

↓

Repeat
```

Notice something.

The loss function tells the model

> "How wrong you are."

Gradient Descent answers

> "Here's how you should improve."

These two concepts always work together.

---

# Chapter 9 — GPT Connection

GPT follows exactly the same process.

Suppose GPT predicts

```text
London
```

instead of

```text
Paris
```

Cross Entropy computes a large loss.

Gradient Descent then updates billions of weights throughout the Transformer.

Each update is extremely small.

But after billions of training examples,

those tiny improvements accumulate.

Eventually,

GPT becomes remarkably good at predicting the next token.

---

# Mental Model

```text
Prediction

↓

Loss

↓

Find Downhill Direction

↓

Small Weight Update

↓

Smaller Loss

↓

Repeat
```

---

# Key Takeaways

✅ The loss function measures how wrong the model is.

✅ Gradient Descent tells the model how to improve.

✅ Learning happens through many small updates.

✅ The learning rate controls the size of each update.

✅ Tiny improvements repeated millions of times produce an intelligent model.

---

# Looking Ahead

Today's lecture explained **what Gradient Descent is**.

But we quietly skipped one crucial question.

> **How does the model know which specific weight should increase or decrease?**

A neural network may contain millions—or even billions—of weights.

How can it compute the contribution of each one?

That question leads us to one of the greatest algorithms in Deep Learning:

**Backpropagation.**