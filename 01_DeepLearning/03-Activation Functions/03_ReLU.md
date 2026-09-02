# Part 3 — ReLU
# The Activation Function That Made Deep Learning Practical

---

# 🎯 Learning Objectives

By the end of this chapter, you should be able to answer:

- Why wasn't Sigmoid enough?
- What is the Vanishing Gradient Problem?
- Why was ReLU invented?
- How does ReLU work?
- Why is ReLU computationally efficient?
- What is Sparse Activation?
- What is the Dying ReLU problem?
- Why do modern LLMs still not use ReLU?

---

# Chapter 20 — Sigmoid Saved Deep Learning... Then Slowed It Down

At first, researchers were thrilled.

The Step Function couldn't learn.

Sigmoid solved that problem beautifully.

Neural networks could finally be trained using Gradient Descent.

Everything seemed perfect.

Then people started building deeper networks.

Not 2 layers.

Not 5 layers.

But 20...

50...

100 layers.

And something strange happened.

The deeper the network became,

the slower it learned.

Sometimes it stopped learning altogether.

---

# Chapter 21 — A Story About Passing a Message

Imagine ten people standing in a line.

The first person whispers

```
100
```

to the next.

Each person is instructed to pass only **10%** of what they hear.

The conversation goes like this.

```
100

↓

10

↓

1

↓

0.1

↓

0.01

↓

0.001
```

By the time the message reaches the last person,

it has almost disappeared.

This is exactly what happens during Backpropagation when using Sigmoid.

Each layer multiplies the gradient by a value less than one.

Layer after layer,

the learning signal becomes smaller and smaller.

Eventually,

the early layers receive almost no information.

This phenomenon is called the

# Vanishing Gradient Problem.

---

# Chapter 22 — Why Does Sigmoid Cause This?

Remember the shape of the Sigmoid curve.

```
          ______

      ___/

 ___/

---------------------------->
```

The curve flattens at both ends.

When the input is very positive,

the output changes only a tiny amount.

When the input is very negative,

the output again changes only a tiny amount.

The slope becomes extremely small.

During Backpropagation,

these tiny slopes are multiplied across many layers.

Tiny × Tiny × Tiny...

Eventually,

the gradient almost vanishes.

The network technically **can** learn,

but it learns painfully slowly.

---

# Chapter 23 — Researchers Asked a Radical Question

What if we stopped trying to make neurons behave like biology?

What if we ignored smooth curves altogether?

What if the activation function were simply...

```
Output = Input
```

for positive values?

Would that still work?

Surprisingly...

Yes.

That simple idea gave birth to one of the most important activation functions ever created.

---

# Chapter 24 — Meet ReLU

ReLU stands for

**Rectified Linear Unit**

Its mathematical definition is incredibly simple.

\[
f(x)=\max(0,x)
\]

Which means

```
If x > 0

Output = x

Otherwise

Output = 0
```

That's it.

No exponential.

No division.

No logarithm.

Just one comparison.

---

# Chapter 25 — Understanding ReLU

Let's evaluate a few inputs.

| Input | Output |
|-------:|-------:|
| -5 | 0 |
| -2 | 0 |
| -0.5 | 0 |
| 0 | 0 |
| 2 | 2 |
| 8 | 8 |
| 15 | 15 |

Notice something interesting.

For positive numbers,

ReLU does absolutely nothing.

It simply lets the value pass through.

For negative numbers,

it completely suppresses them.

---

# Chapter 26 — But Wait... Isn't This Still Linear?

This is one of the most common interview questions.

People look at ReLU and say

> "For positive values,
>
> y = x.
>
> That's just a straight line.
>
> How can it introduce non-linearity?"

Excellent question.

The answer lies in looking at the function as a whole.

The positive region is linear.

The negative region is also linear.

But the entire function is **not** a single straight line.

At zero,

the behaviour changes abruptly.

```
Negative Side

↓

Flat

↓

0

↓

Positive Side

↓

Slope = 1
```

A single straight line cannot behave like this.

The "kink" at zero introduces non-linearity.

Without that bend,

stacking many layers would still collapse into one linear transformation.

Because of that bend,

deep neural networks become capable of learning complex functions.

---

# Chapter 27 — Why ReLU Was a Revolution

ReLU solved two major problems.

### 1. Better Gradient Flow

For every positive input,

the derivative is

\[
1
\]

Not

0.03

Not

0.001

Exactly

1.

This means gradients flow almost unchanged through active neurons.

Deep networks suddenly became much easier to train.

---

### 2. Computational Simplicity

Compare the computations.

Sigmoid needs

- Exponential
- Addition
- Division

ReLU needs only

```
max(0,x)
```

One comparison.

That's all.

When training billions of neurons,

saving even a few operations per neuron becomes enormously important.

---

# Chapter 28 — Sparse Activation

ReLU has another beautiful property.

Imagine a layer containing

1000 neurons.

After applying ReLU,

perhaps only

250 neurons

produce positive outputs.

The remaining

750 neurons

output

```
0
```

Only a small subset of neurons becomes active.

This is called

# Sparse Activation

Instead of every neuron trying to respond,

only the neurons that detect meaningful patterns participate.

This leads to

- Better computational efficiency
- Reduced interference between features
- Better feature specialization

It's surprisingly similar to how biological brains work,

where only a fraction of neurons fire at any given moment.

---

# Chapter 29 — ReLU Isn't Perfect Either

Imagine a neuron whose weighted sum becomes

```
-8
```

ReLU outputs

```
0
```

The derivative is also

```
0
```

Now Gradient Descent tries to update the weights.

But the gradient flowing through this neuron is zero.

The weights barely change.

If this continues for many iterations,

the neuron may never activate again.

It becomes permanently inactive.

This is called the

# Dying ReLU Problem.

The neuron is still physically present in the network,

but from a learning perspective,

it's effectively dead.

---

# Chapter 30 — Why Does This Happen?

Two common reasons are

### Poor Weight Initialization

If the initial weights push most inputs into the negative region,

the neuron starts life inactive.

It may struggle to recover.

---

### Large Learning Rate

Suppose the neuron is active today.

A very large gradient update changes its weights dramatically.

Tomorrow,

all its inputs become negative.

Now the output is always zero.

The gradient becomes zero.

Future updates become almost impossible.

The neuron gets stuck.

This is why choosing an appropriate learning rate is so important.

---

# Chapter 31 — Engineering Perspective

Around 2012,

deep networks trained with ReLU dramatically outperformed earlier models using Sigmoid.

This was one of the key ingredients behind the success of deep convolutional neural networks and helped reignite interest in Deep Learning.

Without ReLU,

training very deep models would have remained much more difficult.

---

# Chapter 32 — Why Doesn't GPT Use ReLU?

If ReLU is faster,

why did GPT switch to GELU?

Because language is subtle.

Consider two activations.

```
-0.01

0.01
```

ReLU treats them very differently.

```
-0.01

↓

0

0.01

↓

0.01
```

A tiny change around zero completely changes the output.

Human language rarely behaves like that.

Words often contribute partially,

not in an all-or-nothing fashion.

GELU handles this more gracefully.

Instead of making a hard decision,

it smoothly scales activations based on their importance.

That makes it a better fit for Transformer models.

We'll study GELU in detail next.

---

# 🎯 Interview Insight

If asked,

> Why did ReLU replace Sigmoid?

A strong answer is:

> "ReLU significantly reduces the Vanishing Gradient Problem because its derivative is 1 for positive inputs, allowing gradients to flow effectively through deep networks. It's also computationally cheaper since it requires only a simple max operation instead of expensive exponential calculations. However, it introduces the Dying ReLU problem, which later motivated smoother activation functions such as GELU."

---

# 📖 Key Takeaways

- Sigmoid enabled learning but suffered from Vanishing Gradients.
- ReLU keeps gradients alive for positive activations.
- ReLU is computationally inexpensive.
- Sparse activation improves efficiency and feature specialization.
- ReLU can suffer from Dying ReLU.
- Modern Transformers prefer GELU because language benefits from smoother activations.

---

# 📖 Next Chapter

ReLU solved the Vanishing Gradient Problem.

But it introduced dead neurons.

Researchers now wondered:

> Can we keep the advantages of ReLU...

> while allowing negative inputs to contribute just a little?

That question led to **Leaky ReLU**.