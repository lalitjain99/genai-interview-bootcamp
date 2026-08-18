# 📘 Lecture — Recurrent Neural Network

> **Central Question:** How can a neural network understand the current element of a sequence while remembering what it has already seen?

---

# 🎬 1. Start With a Simple Sentence

Suppose I give you this sentence one word at a time:

```text
I
```

Then:

```text
I love
```

Then:

```text
I love this
```

Then:

```text
I love this movie
```

When you reach the word:

`movie`

you do not understand it in isolation.

Your understanding is influenced by:

```text
I → love → this
```

You are carrying some information from the past while reading the current word.

That sounds obvious for a human.

But now imagine a normal neural network.

---

# 🧠 2. What Would a Normal Feed-Forward Network Do?

Suppose we process every word independently:

```text
"I"
↓
Neural Network
↓
Output
```

Then:

```text
"love"
↓
Same type of network
↓
Output
```

Then:

```text
"this"
↓
Network
↓
Output
```

The problem is:

> when the network processes `"this"`, where is the information about `"I love"`?

There isn't any built-in mechanism carrying it forward.

The network receives:

> current input

but not:

> previous context.

---

# 🎯 3. Let's Try to Fix That

What if, after processing `"I"`, the network produced some small representation describing what it had understood so far?

Call it:

`h1`

Then when `"love"` arrives, we give the network both:

```text
Current word = "love"

Previous information = h1
```

Now it can produce a new representation:

`h2`

So:

```text
"I"
↓
h1

"love" + h1
↓
h2

"this" + h2
↓
h3

"movie" + h3
↓
h4
```

Now something interesting has happened.

The network has:

> **memory**

Not human memory.

But a numerical representation of previous context.

That representation is called:

# **Hidden State**

---

# 🧠 4. What Exactly Is Hidden State?

Hidden state is simply:

> **a vector that carries learned information from previous sequence positions into the current position.**

For example:

```text
h1 = information after seeing "I"

h2 = information after seeing "I love"

h3 = information after seeing "I love this"

h4 = information after seeing "I love this movie"
```

Do not imagine that:

```text
h3 = ["I", "love", "this"]
```

It is not storing words literally.

Instead:

```text
Entire Past
↓
Learned Compression
↓
Hidden State
```

So `h_t` is better understood as:

> **the model's current understanding of the sequence so far.**

---

# ⭐ 5. We Have Just Invented the Basic RNN Idea

At time step `t`, the model has:

### Current input

`x_t`

### Previous context

`h_{t-1}`

And it needs to produce:

### Updated context

`h_t`

Therefore:

```text
Current Input
+
Previous Hidden State
↓
Neural Transformation
↓
New Hidden State
```

Conceptually:

`h_t = f(x_t, h_{t-1})`

This is the heart of a:

# **Recurrent Neural Network**

---

# 🔄 6. Why Is It Called “Recurrent”?

Because the output state of one step is fed into the next step.

```text
x1
↓
RNN
↓
h1
 │
 └─────────────┐
               ↓
x2 ─────────→ RNN
               ↓
               h2
                │
                └─────────┐
                          ↓
x3 ───────────────────→ RNN
                          ↓
                          h3
```

The hidden state keeps recurring through the network.

Hence:

> **Recurrent Neural Network**

---

# 🧩 7. The Loop Diagram Can Be Misleading

You may see an RNN drawn as:

```text
       ┌─────────────┐
       │             │
       ↓             │
x_t → RNN Cell → h_t ┘
```

This looks like a loop.

But for understanding and training, it is often easier to **unroll it through time**.

---

# 🔓 8. Unrolling the RNN

Suppose sequence is:

```text
I → love → AI
```

The RNN can be visualized as:

```text
h0 ──→ [RNN] ──→ h1 ──→ [RNN] ──→ h2 ──→ [RNN] ──→ h3
        ↑                  ↑                  ↑
       "I"               "love"              "AI"
```

Now it looks like a deep network.

But instead of depth representing:

> different neural-network layers,

it represents:

> different time steps.

This distinction will become very important during BPTT.

---

# 🎯 9. Are These Three Different RNNs?

No.

This is one of the most important RNN concepts.

The diagram:

```text
[RNN] → [RNN] → [RNN]
```

does NOT mean three separately learned networks.

It is:

> **the same RNN cell reused at every time step.**

---

# 🔄 10. Parameter Sharing Across Time

Think back to CNNs.

In a CNN:

```text
Same Filter
↓
Position 1
Position 2
Position 3
...
```

The same weights are reused across:

> space.

RNN does something very similar.

```text
Same Recurrent Transformation
↓
Time 1
Time 2
Time 3
...
```

So:

```text
CNN
→ parameter sharing across SPACE

RNN
→ parameter sharing across TIME
```

This is one of the cleanest bridges between our previous CNN module and RNNs.

---

# 🧠 11. Why Should the Same Weights Be Reused?

Suppose we had separate parameters:

```text
W1 for word 1
W2 for word 2
W3 for word 3
```

Then a sentence of length:

`10`

would require one architecture.

A sentence of length:

`100`

would require another.

That would be terrible.

Instead we want one general rule:

> “Given the current input and my previous state, update my understanding.”

That rule should work at:

```text
time 1
time 2
time 3
...
time T
```

So we learn:

> one recurrent transformation

and reuse it everywhere in time.

---

# 🧮 12. Now the Equation Should Feel Natural

We need to combine two things:

```text
Current Input x_t
+
Previous State h_{t-1}
```

Let's transform the current input:

`W_xh x_t`

and transform the previous state:

`W_hh h_{t-1}`

Then combine them:

`W_xh x_t + W_hh h_{t-1} + b_h`

Finally apply a nonlinearity such as `tanh`.

So:

# `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

This equation should no longer look arbitrary.

It simply means:

> **new memory = nonlinear combination of current information and previous memory**

---

# 🔍 13. Understand Every Term

## `x_t`

Current sequence element.

For text, this could be:

> vector representation of the current token.

---

## `h_{t-1}`

Hidden state from the previous step.

It represents:

> previous context.

---

## `W_xh`

Transforms:

```text
Current Input
→ Hidden-State Space
```

---

## `W_hh`

Transforms:

```text
Previous Hidden State
→ Contribution to New Hidden State
```

This is the recurrent connection.

---

## `b_h`

Bias vector.

---

## `tanh`

Adds nonlinearity and produces the new hidden state.

---

# 🧠 14. A More Intuitive Equation

Instead of initially memorizing:

`h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

first remember:

```text
New Context
=
Current Information
+
Previous Context
+
Learned Transformation
```

The equation is simply the mathematical implementation of that idea.

---

# 🎬 15. Walk Through a Sentence

Let's process:

```text
I → love → AI
```

Assume:

`h0 = 0`

Initially, the network knows nothing about the sequence.

---

# 1️⃣ Time Step 1 — `"I"`

Current input:

`x1 = representation("I")`

Previous state:

`h0 = 0`

So:

`h1 = tanh(W_xh x1 + W_hh h0 + b_h)`

Since `h0` contains no earlier context, `h1` mainly reflects:

> the first token.

Now:

```text
"I"
↓
h1
```

---

# 2️⃣ Time Step 2 — `"love"`

Now:

```text
Current input = x2

Previous state = h1
```

So:

`h2 = tanh(W_xh x2 + W_hh h1 + b_h)`

Now `h2` is influenced by:

```text
"love"
+
information carried from "I"
```

So it represents something closer to:

> the sequence `"I love"`.

---

# 3️⃣ Time Step 3 — `"AI"`

Now:

`h3 = tanh(W_xh x3 + W_hh h2 + b_h)`

But `h2` was already influenced by:

```text
"I" + "love"
```

Therefore `h3` is indirectly influenced by:

```text
"I" → "love" → "AI"
```

This is how recurrence creates:

> sequence context.

---

# ⭐ 16. The Important Recursive Dependency

Notice:

`h3` depends on `h2`.

But:

`h2` depends on `h1`.

And:

`h1` depends on `x1`.

Therefore:

```text
h3
depends on
x3
+
x2
+
x1
```

indirectly.

More generally:

> `h_t` can contain information originating from `x1 ... x_t`.

That is the basic mechanism by which an RNN remembers history.

---

# 🧠 17. But Does `h_t` Store Everything?

No.

This is extremely important.

Suppose hidden size is:

`128`

Whether the sequence has:

```text
5 words
```

or:

```text
500 words
```

the hidden state still has:

`128 numbers`.

So the RNN is continually performing something like:

```text
Past History
+
New Information
↓
Compress
↓
Fixed-Size Hidden State
```

This immediately gives us our first clue that RNNs may struggle with:

> very long sequences.

We'll come back to this.

---

# 📐 18. Hidden State Dimensions

Suppose:

```text
Input vector size D = 300

Hidden size H = 128
```

Then:

`x_t.shape = 300`

`h_t.shape = 128`

Therefore:

`W_xh.shape = 128 × 300`

because:

```text
300-dimensional input
→
128-dimensional hidden state
```

And:

`W_hh.shape = 128 × 128`

because:

```text
128-dimensional previous state
→
128-dimensional new state
```

Bias:

`b_h.shape = 128`

---

# 🧮 19. Vanilla RNN Parameter Count

For:

* input dimension `D`
* hidden size `H`

parameters are:

### Input → Hidden

`H × D`

### Hidden → Hidden

`H × H`

### Bias

`H`

Total:

# `HD + H² + H`

---

# ⭐ 20. Why Sequence Length Doesn't Add Parameters

Suppose:

```text
D = 300
H = 128
```

RNN parameters:

`128×300 + 128×128 + 128`

`= 54,912`

Now sequence length is:

`10`.

Still:

`54,912`

Sequence length becomes:

`1000`.

Still:

`54,912`

Why?

Because:

> the same weights are reused across every time step.

---

# ⚡ 21. But Longer Sequences Still Cost More Compute

The parameters are the same.

But:

```text
10 steps
→ run RNN cell 10 times

1000 steps
→ run RNN cell 1000 times
```

So:

```text
Parameter Count
does not grow with T

Compute
does grow with T
```

Exactly like CNN parameter sharing:

> shared weights reduce parameter count but do not eliminate repeated computation.

---

# 🧠 22. What Is `h0`?

At time `1`, there is no previous hidden state.

So we need:

`h0`

A common choice is:

```text
h0 = [0, 0, ..., 0]
```

Why?

Because before reading the sequence:

> we have no previous context.

---

# ⚠️ 23. Zero Hidden State Is Fine

Do not confuse this with:

> initializing neural-network weights to zero.

`h0` is usually:

> an activation/state.

It is not the recurrent weight matrix itself.

So:

```text
Zero weights
→ can cause symmetry problems

Zero initial hidden state
→ perfectly normal
```

---

# 📤 24. Hidden State vs Output

Another important distinction:

> hidden state is not necessarily the task output.

At each step we may compute:

`y_t = W_hy h_t + b_y`

So:

```text
x_t
↓
RNN
↓
h_t
├────→ carried to next time step
│
└────→ output layer → y_t
```

`h_t` has two possible roles:

1. carry context forward
2. help create a prediction

---

# 🎯 25. What Output Do We Actually Use?

That depends on the task.

Now the sequence-task patterns from the previous lecture become useful.

---

# 🧠 26. Many-to-One

Example:

> sentiment classification

```text
This → movie → was → excellent
  ↓      ↓       ↓        ↓
 h1     h2      h3       h4
                           ↓
                       Classifier
                           ↓
                        Positive
```

We use the final hidden representation:

`h_T`

to make one prediction.

So:

```text
Many Inputs
→
One Output
```

---

# 🏷️ 27. Many-to-Many With Aligned Outputs

Example:

```text
John  lives  in  London
 ↓      ↓     ↓     ↓
PER     O     O    LOC
```

Each hidden state produces an output:

```text
h1 → y1
h2 → y2
h3 → y3
h4 → y4
```

Useful for:

* named entity recognition
* POS tagging
* sequence labeling

---

# 🎯 28. Seq2Seq Will Extend This Idea

Machine translation is harder.

Input:

```text
How are you?
```

Output:

```text
Comment allez-vous ?
```

Input/output lengths can differ.

So later we will use:

```text
Encoder RNN
↓
Input Representation
↓
Decoder RNN
↓
Output Sequence
```

But there is no reason to jump there yet.

First we need to understand:

> how a single RNN learns.

---

# 🧠 29. Now We Encounter the First Real RNN Problem

Look again:

```text
h1
↓
h2
↓
h3
↓
h4
↓
...
↓
h100
```

Suppose something important appeared at:

`x1`.

For `x1` to influence `h100`, that information has to survive:

> 99 recurrent updates.

That is a lot of repeated compression and transformation.

---

# ⏳ 30. Short Dependency vs Long Dependency

Consider:

```text
The sky is dark, so it may ...
```

To predict:

`rain`

recent context may be enough.

That's a:

> short-range dependency.

Now:

```text
I grew up in France. After moving through several
countries and working abroad for decades, the language
I still speak most naturally is ...
```

To predict:

`French`

important information may have appeared:

> much earlier.

That's a:

> long-range dependency.

---

# ⚠️ 31. Hidden State Creates a Bottleneck

Remember:

```text
Entire History
↓
Fixed-Size h_t
```

With every new step:

```text
Old Information
+
New Information
↓
New Fixed-Size State
```

The network has to continually decide implicitly:

> what information survives.

Vanilla RNN has no explicit gates saying:

```text
Keep this.
Forget that.
Remember this for 100 steps.
```

It simply repeatedly applies:

`tanh(W_xh x_t + W_hh h_{t-1} + b)`

This can make long-term memory difficult.

---

# 🧠 32. There Is an Even Deeper Problem During Training

So far we have only looked at:

> forward propagation.

But how will the network learn `W_hh`?

Remember:

> the SAME `W_hh` was used at every time step.

For:

```text
h1 → h2 → h3 → h4
```

the computation graph contains repeated uses of:

`W_hh`.

So when the final prediction is wrong:

> the loss must send gradients backward through these time steps.

---

# 🔄 33. Backward Through Time

Conceptually:

```text
FORWARD

h0
↓
h1
↓
h2
↓
h3
↓
Loss
```

Backward:

```text
Loss
↓
h3
↓
h2
↓
h1
↓
h0
```

This looks just like normal backpropagation through a deep network.

Except:

> depth here came from time.

This technique is called:

# **Backpropagation Through Time — BPTT**

---

# ⭐ 34. Why BPTT Is the Natural Next Lecture

We now have an important question.

The same recurrent weight:

`W_hh`

was used at:

```text
time 1
time 2
time 3
...
```

So:

> how do we calculate one final gradient `dL/dW_hh`?

If you remember CNN backpropagation, you may already suspect the answer.

CNN:

```text
Same filter
used at many spatial locations
↓
Gradient contributions are summed
```

RNN:

```text
Same recurrent matrix
used at many time steps
↓
Gradient contributions are summed
```

But because the hidden states also depend recursively on one another:

> the gradient story becomes much more interesting.

That is exactly what BPTT explains.

---

# ⚠️ 35. And BPTT Will Reveal Another Problem

When gradients travel from:

```text
h100
↓
h99
↓
h98
↓
...
↓
h1
```

they repeatedly encounter:

* recurrent weight matrix
* activation derivatives

This involves repeated multiplication.

Depending on the values:

```text
gradient
→ smaller
→ smaller
→ smaller
→ almost zero
```

or:

```text
gradient
→ larger
→ larger
→ enormous
```

These are:

# **Vanishing Gradients**

and

# **Exploding Gradients**

---

# 🧠 36. And That Will Naturally Lead Us to LSTM

So the progression is not arbitrary.

It is:

```text
Need sequence memory
↓
RNN
↓
Hidden state
↓
Repeated recurrent transformation
↓
Train with BPTT
↓
Long gradient paths
↓
Vanishing / exploding gradients
↓
Poor long-term dependency learning
↓
Need better memory mechanism
↓
LSTM / GRU
```

That is the story we will follow.

---

# 🔄 37. Why RNNs Are Sequential

There is one more important consequence of:

`h_t = f(x_t, h_{t-1})`

To calculate:

`h2`

we need:

`h1`.

To calculate:

`h3`

we need:

`h2`.

Therefore:

```text
h1 → h2 → h3 → h4
```

cannot be fully computed simultaneously.

This gives RNNs:

> inherently sequential processing across time.

---

# ⚡ 38. Batch Parallelism vs Time Parallelism

Suppose we have:

`32 sequences`

We can process the 32 examples of a time step together.

So RNNs can use:

> batch parallelism.

But within each sequence:

```text
t1 → t2 → t3
```

must remain ordered.

So:

```text
Across Batch
→ parallel

Across Time
→ sequential
```

This later becomes another important reason:

> Transformers scale better during training.

---

# 🌊 39. A Natural Strength of RNNs — Streaming

Sequential processing is not always bad.

Imagine a sensor:

```text
x1 arrives
↓
update h1

x2 arrives
↓
update h2

x3 arrives
↓
update h3
```

You can keep:

> the current hidden state

and update it when new data arrives.

You do not necessarily need to reprocess the entire history.

This makes recurrence conceptually attractive for:

> streaming data.

---

# 🎯 40. RNN vs Feed-Forward Network

| Feed-Forward Network                   | RNN                               |
| -------------------------------------- | --------------------------------- |
| No recurrent state                     | Hidden state                      |
| Processes fixed representation         | Processes ordered sequence        |
| No built-in history                    | Carries previous context          |
| Different positions not naturally tied | Same parameters across time       |
| Easy parallel processing               | Sequential dependency across time |

---

# 🔗 41. RNN vs CNN

| CNN                                 | RNN                         |
| ----------------------------------- | --------------------------- |
| Spatial data                        | Sequential data             |
| Shares weights across space         | Shares weights across time  |
| Local spatial receptive field       | Running temporal context    |
| Feature maps                        | Hidden states               |
| Spatial computation highly parallel | Time computation sequential |

A useful mental shortcut:

```text
CNN
→ What pattern exists around HERE?

RNN
→ Given what I knew BEFORE, what do I know NOW?
```

---

# ⚠️ 42. Common Misconceptions

### ❌ Hidden state contains the entire past exactly

No.

It is:

> a learned compressed representation.

---

### ❌ Every time step has different weights

No.

Standard RNN shares:

> the same recurrent parameters.

---

### ❌ `h_t` is always the task output

No.

It is an internal state that may be used to produce an output.

---

### ❌ RNN completely solves long-term memory

No.

Vanilla RNNs struggle significantly with long dependencies.

---

### ❌ Sequence length increases RNN parameter count

No.

It increases:

> repeated computation.

---

### ❌ RNN time steps can all be calculated simultaneously

Not in a standard recurrent formulation because:

`h_t` depends on `h_{t-1}`.

---

# 🧮 43. Core Equation Sheet

## Pre-Activation

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

## Hidden State

`h_t = tanh(a_t)`

## Optional Output

`y_t = W_hy h_t + b_y`

## Recurrent Parameter Count

`HD + H² + H`

where:

* `D` = input dimension
* `H` = hidden size

---

# 🎤 44. Interview Answer — What Is an RNN?

> **An RNN is a neural network designed for sequential data. At each time step it combines the current input with a hidden state carried from the previous time step to produce a new hidden state. The same recurrent parameters are reused across all positions, allowing the network to model ordered dependencies while keeping parameter count independent of sequence length.**

---

# 🎤 45. Interview Answer — What Is Hidden State?

> **The hidden state is a learned fixed-dimensional representation of the sequence context seen so far. It is passed from one time step to the next and allows the current computation to depend indirectly on earlier elements. It should be thought of as compressed context rather than an exact memory of every previous token.**

---

# 🎤 46. Interview Answer — Why Are RNN Weights Shared?

> **The same transformation should be applicable at every sequence position. Sharing recurrent parameters keeps parameter count independent of sequence length, allows variable-length processing, and gives the network one consistent rule for updating its state across time.**

---

# 🧠 47. The RNN Story in One Diagram

```text
Problem:
Current meaning depends on previous context
↓
Need Memory
↓
Represent Memory as Hidden State
↓
Current Input + Previous Hidden State
↓
Shared Recurrent Transformation
↓
New Hidden State
↓
Carry Forward
↓
Repeat Across Sequence
```

But then:

```text
Long Sequence
↓
Information Must Survive Many Updates
↓
Training Gradient Must Travel Through Many Steps
↓
BPTT
↓
Vanishing / Exploding Gradient Problem
↓
LSTM / GRU
```

---

# ⭐ 48. Golden Mental Model

Do not start by memorizing:

`h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)`

Start with:

```text
What do I see NOW?
+
What do I remember from BEFORE?
↓
Update my understanding
↓
Carry it FORWARD
```

That is an RNN.

The equation is simply:

> the mathematical implementation of that story.

---

# 🚀 Where We Go Next

Now we understand:

> how an RNN moves information forward through a sequence.

Our next question is:

> **How does the learning signal move backward through those same time steps?**

That takes us naturally to:

# `03_Backpropagation_Through_Time`

where we will derive BPTT using the same chain-rule thinking we already learned in neural networks and CNN backpropagation.

---

# ⭐ Golden Rule

> **An RNN is a neural network that repeatedly asks: “Given the current input and everything my hidden state remembers from the past, what should my new state be?”**
