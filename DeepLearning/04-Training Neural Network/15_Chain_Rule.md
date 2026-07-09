# Chain Rule — How Errors Travel Backward Through a Neural Network

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- Why do we need the Chain Rule?
- What problem does the Chain Rule solve?
- What does "a function of a function" mean?
- Why can't we compute gradients directly?
- How does the Chain Rule make Backpropagation possible?
- Why is the Chain Rule called the backbone of Backpropagation?

---

# Chapter 1 — The Missing Piece

In the previous lecture, we learned that Backpropagation computes the gradient for every weight.

But we left one important question unanswered.

> **How does Backpropagation actually calculate those gradients?**

Imagine a very simple neural network.

```text
Input
   │
   ▼
Weighted Sum (z)
   │
   ▼
Activation (a)
   │
   ▼
Loss (L)
```

Notice something.

The weight **does not affect the loss directly**.

Instead,

the weight affects **z**,

which affects **a**,

which finally affects the **loss**.

The computation happens in stages.

This creates a new challenge.

---

# Chapter 2 — A Domino Effect

Imagine a row of dominoes.

```text
Domino A

↓

Domino B

↓

Domino C

↓

Domino D
```

If you push Domino A,

Domino B falls.

Because Domino B falls,

Domino C falls.

Eventually,

Domino D falls.

Now suppose someone asks,

> **How much did Domino A contribute to Domino D falling?**

You cannot answer by looking only at Domino A or only at Domino D.

You must follow the entire chain.

The same thing happens inside a neural network.

A weight influences the loss through many intermediate computations.

---

# Chapter 3 — Functions Inside Functions

Suppose we define

```text
y = f(x)
```

Easy.

Now suppose

```text
z = g(y)
```

Now,

the value of **z** depends on **y**,

and **y** depends on **x**.

We can rewrite this as

```text
z = g(f(x))
```

This is called a

> **Function of a Function**

In neural networks,

this happens continuously.

Each layer becomes the input to the next layer.

---

# Chapter 4 — A Neural Network Is One Giant Nested Function

A neural network is not one equation.

It is many equations connected together.

```text
Input

↓

Linear Layer

↓

ReLU

↓

Linear Layer

↓

Softmax

↓

Cross Entropy Loss
```

Each step depends on the previous step.

The loss therefore depends on

- Softmax
- Softmax depends on logits
- Logits depend on weights
- Weights depend on the previous layer
- Previous layer depends on earlier weights

Everything is connected.

---

# Chapter 5 — The Need for the Chain Rule

Suppose we want to answer

> **How much does changing a weight change the final loss?**

The weight does not directly touch the loss.

Instead,

it changes

```text
Weight

↓

z

↓

Activation

↓

Loss
```

So we must combine

- how the weight changes **z**
- how **z** changes the activation
- how the activation changes the loss

Instead of treating these separately,

mathematics provides a beautiful shortcut.

This shortcut is called the **Chain Rule**.

---

# Chapter 6 — The Chain Rule

Suppose

$$
z = g(y)
$$

and

$$
y = f(x)
$$

Then

$$
\frac{dz}{dx}
=
\frac{dz}{dy}
\times
\frac{dy}{dx}
$$

This equation says something surprisingly intuitive.

> **The total effect is the product of all the individual effects along the path.**

Think back to the dominoes.

Every domino passes the effect to the next one.

The final effect depends on every link in the chain.

---

# Chapter 7 — Applying the Chain Rule to a Neural Network

Consider the simplest possible network.

```text
w

↓

z = wx + b

↓

a = ReLU(z)

↓

Loss
```

Suppose we want to compute

$$
\frac{\partial L}{\partial w}
$$

The weight does not directly influence the loss.

Instead,

```text
w

↓

z

↓

a

↓

L
```

Using the Chain Rule,

we write

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\times
\frac{\partial a}{\partial z}
\times
\frac{\partial z}{\partial w}
$$

Notice what happened.

Instead of solving one difficult problem,

we broke it into three much simpler problems.

This is exactly how Backpropagation computes gradients.

---

# Chapter 8 — Why Does the Chain Rule Use Multiplication Instead of Addition?

One question naturally arises after seeing the Chain Rule.

We write

$$
\frac{dL}{dw}
=
\frac{dL}{da}
\times
\frac{da}{dz}
\times
\frac{dz}{dw}
$$

But why do we **multiply** these quantities?

Why not add them?

This is a very important question.

To answer it, let's forget neural networks for a moment.

---

## A Manufacturing Analogy

Imagine a factory that manufactures T-shirts.

The production happens in two stages.

```text
Cotton

↓

Cloth

↓

T-Shirts
```

Now suppose we know the following.

### Stage 1

Every

```text
1 kg Cotton
```

produces

```text
5 meters of Cloth
```

So we can say

```text
1 Cotton

↓

5 Cloth
```

---

### Stage 2

Now suppose

```text
1 meter Cloth
```

produces

```text
3 T-Shirts
```

So we have

```text
1 Cloth

↓

3 Shirts
```

---

## The Final Question

Now someone asks

> **If I have one extra kilogram of cotton, how many extra T-shirts can I produce?**

Let's follow the chain.

```text
1 Cotton

↓

5 Cloth

↓

15 Shirts
```

How did we obtain

```text
15 Shirts
```

Simple.

We multiplied.

```text
5 × 3 = 15
```

Not added.

---

## Why Multiplication?

Think carefully about what is happening.

One kilogram of cotton produces

```text
5 Cloth
```

Now,

**each one** of those five pieces of cloth produces

```text
3 Shirts
```

So the second effect happens **for every result produced by the first effect**.

That is why we multiply.

---

## Why Doesn't Addition Work?

Suppose we tried adding.

```text
5 + 3 = 8
```

What does

```text
8
```

represent?

Nothing.

It has no physical meaning.

The two numbers describe completely different relationships.

- **5 Cloth per Cotton**
- **3 Shirts per Cloth**

Adding them mixes unrelated quantities.

---

## Understanding the Units

Let's write the units.

```text
5 Cloth / Cotton

×

3 Shirts / Cloth
```

Now multiply them.

```text
5 Cloth × 3 Shirts
-------------------
Cotton × Cloth
```

The word **Cloth** appears in both the numerator and denominator.

It cancels.

Leaving us with

```text
15 Shirts / Cotton
```

Exactly the quantity we wanted.

---

## Applying This to Neural Networks

Now replace the factory with a neural network.

```text
Weight

↓

Weighted Sum

↓

Activation

↓

Loss
```

Suppose changing the weight by **1 unit**

changes the weighted sum by

```text
4 units
```

Next,

every **1 unit** change in the weighted sum

changes the activation by

```text
2 units
```

Finally,

every **1 unit** change in the activation

changes the loss by

```text
5 units
```

Let's follow the chain.

```text
Weight

↓

4 Weighted Sum

↓

8 Activation

↓

40 Loss
```

Where did **40** come from?

```text
4 × 2 × 5
```

Exactly the same logic as the factory example.

---

## Looking at the Units Again

Each derivative is actually a **rate of change**.

For example,

$$
\frac{dL}{da}
$$

means

```text
Loss

per

Activation
```

Similarly,

$$
\frac{da}{dz}
$$

means

```text
Activation

per

Weighted Sum
```

Multiplying them gives

$$
\frac{Loss}{Activation}
\times
\frac{Activation}{WeightedSum}
=
\frac{Loss}{WeightedSum}
$$

Notice that **Activation** cancels automatically.

Now multiply again.

$$
\frac{Loss}{WeightedSum}
\times
\frac{WeightedSum}{Weight}
=
\frac{Loss}{Weight}
$$

The **Weighted Sum** also cancels.

We are left with

$$
\frac{Loss}{Weight}
$$

which is exactly the quantity we wanted.

---

## The Beautiful Intuition

Every arrow inside a neural network answers one small question.

```text
Weight
      │
      ▼
Weighted Sum
      │
      ▼
Activation
      │
      ▼
Loss
```

- How does the weight affect the weighted sum?
- How does the weighted sum affect the activation?
- How does the activation affect the loss?

The Chain Rule simply combines these **local effects** into one **global effect**.

That is why the Chain Rule uses **multiplication**.

Every intermediate quantity cancels naturally,

leaving only the overall effect of changing the weight on the final loss.

---

# Interview Insight

A common interview question is:

> **Why does the Chain Rule multiply derivatives instead of adding them?**

The answer is:

Each derivative represents a **local rate of change**.

To compute the total effect across multiple stages, we compose these local rates by multiplying them. Multiplication allows the intermediate quantities to cancel, producing the final rate of change that we are interested in. Addition cannot do this because the derivatives describe different relationships and have different units.

---

# Chapter 9 — Why Backpropagation Is Efficient

Imagine a network with hundreds of layers.

Without the Chain Rule,

every gradient would require solving a huge equation from scratch.

With the Chain Rule,

each layer only needs to answer one question.

> **How does my output affect the next layer?**

Each layer passes this information backward,

allowing gradients to flow efficiently through the network.

This is why Backpropagation scales to networks containing billions of parameters.

---

# Chapter 10 — Connecting Everything

Let's connect all the ideas we've learned so far.

```text
Training Data
        │
        ▼
Forward Pass
        │
        ▼
Prediction
        │
        ▼
Loss Function
        │
        ▼
Backpropagation
        │
        ▼
Chain Rule
        │
        ▼
Partial Derivatives
        │
        ▼
Gradient Vector
        │
        ▼
Gradient Descent
        │
        ▼
Updated Weights
```

Every concept we've studied now fits into one complete learning pipeline.

---

# Looking Ahead

The Chain Rule explains the mathematics behind Backpropagation.

But modern Deep Learning frameworks like **PyTorch** and **TensorFlow** don't manually apply the Chain Rule for every operation.

Instead,

they represent every computation as a **Computational Graph**.

The graph automatically tracks every operation,

and during Backpropagation,

it automatically applies the Chain Rule.

That is exactly what we'll study in the next lecture.

---

# Key Takeaways

- Neural networks are composed of many functions connected together.
- The loss depends on the weights through many intermediate computations.
- The Chain Rule computes the total effect by multiplying the local effects at each step.
- Backpropagation is essentially the repeated application of the Chain Rule.
- Without the Chain Rule, efficient training of deep neural networks would not be possible.
- Modern frameworks automate the Chain Rule using Computational Graphs.


---

# Deep Dive — Why Do We Differentiate Only with Respect to Weights?

A thoughtful question naturally arises at this point.

Earlier, we said that the loss depends on

- Softmax
- Logits
- Activations
- Weighted Sums
- Weights

But what about the **input**?

After all, without the input, none of these quantities would even exist.

So why don't we compute gradients with respect to the input during training?

---

## Understanding the Root Variables

Consider a simple neural network.

```text
Input (x)
      │
      ▼
z₁ = w₁x + b₁
      │
      ▼
a₁ = ReLU(z₁)
      │
      ▼
z₂ = w₂a₁ + b₂
      │
      ▼
Softmax
      │
      ▼
Loss
```

If we look carefully,

there are only **three independent variables** in this entire computation.

```text
Input (x)

Weights (w)

Biases (b)
```

Everything else is **computed** from these.

For example,

```text
Input
      │
      ▼
Weighted Sum (z)
      │
      ▼
Activation (a)
      │
      ▼
Logits
      │
      ▼
Softmax Probabilities
      │
      ▼
Loss
```

Notice that

- Weighted sums are computed.
- Activations are computed.
- Logits are computed.
- Probabilities are computed.
- Even the loss is computed.

These are **derived quantities**.

They are not parameters that the model learns.

---

## During Training, What Can We Actually Change?

Suppose our training example is

```text
Image = Cat
```

Can we change the image?

No.

The dataset is fixed.

The image is simply given to the network.

The only quantities that we are allowed to modify are

```text
Weights

Biases
```

These represent the knowledge stored inside the neural network.

Therefore, during training, our real question becomes

> **How should the weights and biases change so that the final loss becomes smaller?**

That is why Backpropagation computes

$$
\frac{\partial L}{\partial w}
\quad\text{and}\quad
\frac{\partial L}{\partial b}
$$

rather than updating the input.

---

## An Analogy — Baking a Cake

Imagine you're baking a cake.

The final taste depends on the ingredients.

```text
Flour

Sugar

Butter

Eggs
```

During baking,

many intermediate things happen.

```text
Ingredients

↓

Mixing

↓

Batter

↓

Baking

↓

Cooling

↓

Cake
```

Suppose the cake doesn't taste good.

Would you try to modify the batter after the cake has already been baked?

Of course not.

Instead,

you adjust the ingredients.

The ingredients are the **independent variables**.

Everything else is simply the result of those ingredients.

A neural network works exactly the same way.

- **Inputs** come from the dataset.
- **Weights and biases** are the learnable ingredients.
- **Weighted sums, activations, logits, probabilities, and loss** are all derived quantities.

By changing the weights and biases,

all of the downstream computations automatically change,

which ultimately changes the final loss.

---

## Key Insight

A neural network has three independent variables:

- **Input**
- **Weights**
- **Biases**

Everything else—weighted sums, activations, logits, Softmax probabilities, and the loss—is a **derived quantity** computed during the forward pass.

During training, the input is fixed by the dataset.

Therefore, the model learns by updating only the **weights** and **biases**, while all the derived quantities automatically change as a consequence.