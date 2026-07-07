# Part 2 — Sigmoid
# The First Trainable Activation Function

---

# 🎯 Learning Objectives

By the end of this chapter you should be able to answer:

- Why did the Step Function fail?
- What problem was Sigmoid trying to solve?
- What makes Sigmoid differentiable?
- Why did it become the most popular activation function?
- Why does Sigmoid output probabilities?
- Why does Sigmoid suffer from Vanishing Gradients?
- Why don't modern LLMs use Sigmoid?

---

# Chapter 10 — AI Winter

The year is now 1969.

Researchers had spent nearly a decade experimenting with Perceptrons.

The excitement was enormous.

Many believed intelligent machines were only a few years away.

Then everything changed.

Marvin Minsky and Seymour Papert published a book called

> **Perceptrons**

The book mathematically proved something devastating.

A single Perceptron could never solve problems like XOR.

Worse...

Researchers realized another problem.

Even if we built larger neural networks,

there was still no practical way to train them.

The culprit?

The Step Function.

---

# Chapter 11 — The Problem Wasn't Intelligence

Let's think carefully.

Suppose the network predicts

```
Dog
```

The correct answer is

```
Cat
```

Obviously the model is wrong.

The obvious solution is

> Change the weights.

Simple enough.

But another question appears.

**Which weight?**

By how much?

Increase?

Decrease?

How do we know?

Imagine trying to improve a recipe.

```
Tea

Milk

Sugar

Cardamom
```

The tea tastes too sweet.

Should you reduce milk?

Increase tea leaves?

Reduce sugar?

Without knowing which ingredient caused the problem,

you're simply guessing.

Neural Networks faced the exact same challenge.

---

# Chapter 12 — A Tiny Change

Researchers realised something profound.

Suppose increasing a weight by a tiny amount causes

the output to increase only slightly.

Good.

Now we know that weight has some influence.

Suppose another weight causes

a huge change.

Even better.

Now we know which parameter matters more.

In other words,

we don't just need predictions.

We need to know

> **How sensitive is the output to small changes?**

That single question gave birth to one of the most important ideas in Machine Learning.

**Derivatives.**

---

# Chapter 13 — Why Step Function Could Never Learn

Let's look at the graph mentally.

```
        1
        │──────────────
        │
────────┼──────────────
        │
        │
        0
```

Imagine standing on the flat part.

You take one tiny step.

Does the output change?

No.

Take another step.

Still no change.

The slope is

```
0
```

Now imagine standing exactly at the jump.

```
0

↓

1
```

The graph changes infinitely fast.

The slope doesn't even exist.

So the derivative of the Step Function is

```
0

almost everywhere

Undefined

at one point.
```

This creates a disaster.

Gradient Descent asks

> Which direction should I move?

The Step Function replies

> I have no idea.

Learning stops.

---

# Chapter 14 — What Would the Perfect Activation Function Look Like?

Researchers now had a wishlist.

The new activation function should

✅ Behave like a Step Function.

✅ Still make binary decisions.

✅ Be smooth.

✅ Be differentiable.

✅ Tell us how the output changes.

Can we build such a function?

Yes.

Meet the Sigmoid Function.

---

# Chapter 15 — Enter Sigmoid

Mathematically,

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

At first glance,

it looks much more complicated.

But don't panic.

Let's understand what it actually does.

Instead of making a hard decision,

Sigmoid makes a **soft decision**.

---

Suppose

```
Input = 100
```

Sigmoid outputs

```
0.99995
```

Very confident.

---

Suppose

```
Input = -100
```

Output becomes

```
0.00004
```

Again,

very confident.

---

Suppose

```
Input = 0
```

Output becomes

```
0.5
```

Interesting.

The model is saying

> "I'm equally uncertain."

Notice something beautiful.

Instead of producing

```
0

or

1
```

Sigmoid produces

every value between

```
0

and

1
```

---

# Chapter 16 — Why Does Sigmoid Look Like a Probability?

This is where many interview candidates get confused.

People often say

> Sigmoid outputs probabilities.

That's only partially correct.

Sigmoid outputs a value between

```
0

and

1
```

Because of this range,

we can interpret the output

as a confidence score.

Example

```
0.98
```

means

```
98% confidence
```

that the input belongs to the positive class.

It doesn't magically understand probability.

Its mathematical range simply makes this interpretation possible.

This is exactly why Sigmoid became popular for

Binary Classification.

---

# Chapter 17 — The Biggest Advantage

Remember the Step Function?

Its graph looked like a cliff.

Sigmoid transforms that cliff

into a smooth hill.

Now imagine standing anywhere on that curve.

Every small movement changes the output

a little.

That means

the derivative exists.

Gradient Descent finally has guidance.

Instead of

```
Move?

Don't know.
```

The derivative now says

```
Move left.

A little.

Now stop.

Now move right.

Tiny amount.
```

Learning finally became possible.

This was one of the biggest breakthroughs in Deep Learning.

---

# Chapter 18 — The Hidden Trap

Researchers celebrated.

Neural Networks could finally learn.

But after building deeper networks,

another problem appeared.

Imagine the output of Sigmoid is

```
0.1
```

During Backpropagation,

gradients multiply across many layers.

```
0.1

×

0.1

×

0.1

×

0.1
```

Very quickly,

the gradient becomes

```
0.000001
```

Almost zero.

Weight updates become microscopic.

Learning almost stops.

This became known as the

# Vanishing Gradient Problem.

Ironically,

the function invented to make learning possible

eventually slowed learning in deep networks.

---

# Chapter 19 — Engineering Perspective

Suppose GPT had 96 Transformer layers

and every Feed Forward Network used Sigmoid.

During Backpropagation,

gradients would shrink layer after layer.

The earliest layers would barely learn.

Training would become extremely slow,

sometimes practically impossible.

This is one of the biggest reasons

modern architectures replaced Sigmoid

with ReLU

and later

GELU.

---

# 🎯 Interview Insight

If an interviewer asks

> "Why did Sigmoid replace the Step Function?"

A strong answer is

> "The Step Function introduced binary decision making but was not differentiable, making Gradient Descent impossible. Sigmoid provided a smooth approximation of the Step Function, allowing derivatives to exist everywhere. This enabled gradient-based optimization and made multi-layer neural networks trainable."

---

# 🧠 Common Interview Mistakes

❌ "Sigmoid outputs probabilities."

Better answer

> Sigmoid outputs values between 0 and 1 that are commonly interpreted as probabilities.

---

❌ "Sigmoid solved Deep Learning."

Better answer

> Sigmoid made gradient-based learning possible but later introduced the Vanishing Gradient Problem, motivating newer activation functions like ReLU and GELU.

---

# 📖 Key Takeaways

- Step Function could make decisions but couldn't learn.
- Sigmoid is a smooth approximation of the Step Function.
- Sigmoid is differentiable.
- It outputs values between 0 and 1.
- These outputs can be interpreted as confidence scores.
- Sigmoid enabled Gradient Descent.
- Deep networks exposed the Vanishing Gradient Problem.
- Modern LLMs therefore prefer GELU over Sigmoid.

---

# 📖 Next Chapter

Sigmoid solved one problem...

but created another.

Researchers now asked

> Can we build an activation function that keeps gradients alive even in very deep neural networks?

That question gave birth to one of the most influential activation functions in history.

**ReLU.**