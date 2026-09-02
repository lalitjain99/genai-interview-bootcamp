# The Idea of Derivatives — Measuring How Things Change

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- Why were derivatives invented?
- What problem do derivatives solve?
- What is the difference between average change and instantaneous change?
- What does a derivative actually measure?
- Why are derivatives essential in Deep Learning?
- Why can't Gradient Descent work without derivatives?

---

# Chapter 1 — A Simple Question

Imagine you're driving from Delhi to Agra.

After one hour, you've travelled

```text
60 km
```

Now someone asks you,

> **"How fast were you driving?"**

Can you answer the question?

Not exactly.

Knowing only the distance isn't enough.

To determine your speed, we also need to know **how much time** it took.

Suppose it took exactly one hour.

Then,

```text
Speed = Distance / Time

      = 60 km / 1 hour

      = 60 km/h
```

Notice something interesting.

We didn't just calculate a distance.

We calculated **how quickly the distance changed with time**.

This idea appears everywhere in our daily lives.

---

# Chapter 2 — The World Is Full of Change

Think about the following questions.

- How fast is a car moving?
- How quickly is a child's height increasing?
- How rapidly is the temperature rising?
- How fast are company profits growing?
- How quickly is a stock price falling?

Although these questions seem different,

they all ask the same thing.

> **How fast is one quantity changing with respect to another?**

Mathematicians call this the **rate of change**.

---

# Chapter 3 — Bringing This Idea to Neural Networks

Now let's return to Deep Learning.

Earlier, we learned about **Loss Functions**.

Suppose our neural network has a weight

```text
w = 2.5
```

The corresponding loss is

```text
Loss = 0.82
```

Now imagine we slightly increase the weight.

```text
w = 2.6
```

The loss becomes

```text
Loss = 0.75
```

Interesting!

A tiny change in the weight caused the loss to decrease.

Now imagine changing the weight again.

```text
w = 2.7
```

The loss becomes

```text
0.70
```

Clearly,

changing the weight changes the loss.

This raises an important question.

> **How sensitive is the loss to changes in the weight?**

This is exactly the kind of question derivatives were invented to answer.

---

# Chapter 4 — A Mountain Adventure

Imagine you're hiking on a mountain.

```
                Peak
                 ▲
                /\
               /  \
              /
             X
```

You are standing at point **X**.

You want to know

> "If I take one tiny step forward, how steep is the mountain right here?"

Notice the words

**right here**.

You don't care about the average steepness of the entire mountain.

You only care about the steepness at your current position.

That is precisely what a derivative measures.

---

# Chapter 5 — Average Rate of Change

Let's first solve an easier problem.

Suppose we have two points.

```
A -------------------- B
```

Point A

```text
Height = 100 m
```

Point B

```text
Height = 140 m
```

The horizontal distance between them is

```text
20 m
```

The average slope is

```text
Rise
──────
Run
```

or

```text
(140 − 100)

────────────

20

=

2
```

This tells us that, on average,

the mountain rises **2 meters** for every **1 meter** we move horizontally.

This is called the **average rate of change**.

---

# Chapter 6 — The Problem with Average Slope

Real mountains aren't straight.

They curve.

```
          *
       *
    *
  *
*
```

Now the average slope depends on

which two points we choose.

If we pick points far apart,

we get one answer.

If we pick different points,

we get another answer.

But we don't want an average.

We want to know

> **How steep is the mountain exactly where I'm standing?**

Average slope can no longer answer that question.

We need something better.

---

# Chapter 7 — The Brilliant Idea

Imagine taking two points on the curve.

```
●--------------------●
```

Now move them closer.

```
●-----------●
```

Closer still.

```
●-----●
```

Even closer.

```
●-●
```

Eventually,

the two points become almost the same point.

Now the average slope becomes

the slope **at that exact point**.

This is called the **instantaneous rate of change**.

And this is exactly what a **derivative** measures.

---

# Chapter 8 — The Mathematical Definition

Mathematicians represent the derivative as

$$
\frac{dy}{dx}
$$

or

$$
\frac{df(x)}{dx}
$$

Don't let the notation intimidate you.

It simply means

> **How much does \(y\) change when \(x\) changes by a tiny amount?**

That's all a derivative is.

It measures the **instantaneous rate of change**.

---

# Chapter 9 — Connecting This to Gradient Descent

Now replace

```text
y
```

with

```text
Loss
```

and replace

```text
x
```

with

```text
Weight
```

Now our question becomes

> **How much does the loss change if I slightly change a weight?**

Mathematically,

```text
Loss = f(weight)
```

The derivative tells us

```text
d(Loss)
────────
d(weight)
```

If this value is

positive,

increasing the weight increases the loss.

If it is

negative,

increasing the weight decreases the loss.

This is exactly the information Gradient Descent needs.

Without derivatives,

Gradient Descent has no idea which direction to move.

---

# Chapter 10 — Looking Ahead

So far,

we've discussed only **one weight**.

Real neural networks don't have one weight.

They have

```text
Millions of weights.
```

Each weight influences the loss differently.

This raises a new question.

> **How do we measure the effect of changing one weight while keeping all the others fixed?**

To answer that,

we need a new concept.

**Partial Derivatives.**

That is exactly what we'll study in the next lecture.

---

# Key Takeaways

- A derivative measures **how quickly one quantity changes with respect to another**.
- It represents the **instantaneous rate of change**, not the average rate of change.
- Average slope uses two distant points.
- A derivative measures the slope at a single point.
- In Deep Learning, derivatives tell us how sensitive the loss is to changes in the weights.
- Gradient Descent relies on derivatives to decide which direction to update the weights.