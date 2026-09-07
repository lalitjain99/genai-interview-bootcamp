# 📘 Why Transformers Were Invented

## *The Moment Attention Stopped Being an Add-On and Became the Architecture*

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to answer:

* What problem had Attention already solved?
* What important problem still remained after Attention?
* Why was recurrence still a limitation?
* Why didn't LSTM or GRU solve that limitation?
* How does an RNN move information between distant words?
* Why are long dependency paths undesirable?
* What important clue did classical Attention give researchers?
* What happens if tokens are allowed to communicate directly?
* Why does removing recurrence improve training parallelism?
* What new problem appears when recurrence is removed?
* What fundamental idea led to the Transformer?
* Why was the Transformer a natural next step rather than a completely unrelated invention?

---

# 🌍 Chapter 1 — Where We Left Off

Let's continue exactly where our Attention module ended.

Before Attention, Seq2Seq looked like this:

```text
Source Sequence
      ↓
Encoder
      ↓
One Fixed Context Vector
      ↓
Decoder
```

The entire source sequence had to be compressed into:

```text
one vector
```

That was the **Context Vector Bottleneck**.

Then Attention changed everything.

Instead of throwing away the intermediate encoder states:

```text
h₁
h₂
h₃
...
hₙ
```

we kept all of them.

At every decoder timestep, the decoder could ask:

> Which encoder states contain the information I need right now?

So instead of one fixed context:

```text
c
```

we created a dynamic context:

```text
c₁
c₂
c₃
...
```

using:

```text
c_t = Σᵢ α_(t,i) hᵢ
```

This solved a major problem.

The decoder now had:

# **Dynamic Source Access**

---

# 🤔 But Wait...

Our architecture still looked like this:

```text
Source
   ↓
RNN / LSTM Encoder
   ↓
h₁  h₂  h₃ ... hₙ
       ↑
    Attention
       ↑
RNN / LSTM Decoder
```

Attention had improved the communication between:

```text
Decoder
↕
Encoder
```

But something fundamental had **not changed**.

The encoder was still recurrent.

The decoder was still recurrent.

So before moving forward, we need to ask:

> **What is still wrong with recurrence?**

---

# 🌍 Chapter 2 — A Simple Sentence

Consider this sentence:

> **The animal didn't cross the street because it was tired.**

Focus on the word:

```text
it
```

To understand the meaning of:

```text
it
```

we need to know what it refers to.

In this sentence:

```text
it
↓
animal
```

So the representation of `it` should somehow use information from:

```text
animal
```

No problem.

Our RNN can do that.

But let's look at **how**.

---

# 🌍 Chapter 3 — How an RNN Moves Information

Suppose the words are processed from left to right.

```text
The
 ↓
animal
 ↓
didn't
 ↓
cross
 ↓
the
 ↓
street
 ↓
because
 ↓
it
 ↓
was
 ↓
tired
```

Recall:

```text
h_t = f(x_t, h_(t-1))
```

The hidden state for:

```text
animal
```

is passed into the next timestep.

That information can then flow into:

```text
didn't
```

then:

```text
cross
```

then:

```text
the
```

then:

```text
street
```

then:

```text
because
```

and finally reach:

```text
it
```

Conceptually:

```text
animal
   ↓
didn't
   ↓
cross
   ↓
the
   ↓
street
   ↓
because
   ↓
it
```

The information can reach `it`.

But notice something.

It had to travel through:

```text
many intermediate states
```

before getting there.

---

# 🧠 Chapter 4 — The Real Problem Is the Path

Imagine the important words are much farther apart.

```text
word 1
```

needs to influence:

```text
word 100
```

In an RNN:

```text
word 1
   ↓
word 2
   ↓
word 3
   ↓
word 4
   ↓
...
   ↓
word 100
```

The information passes through almost one hundred recurrent transitions.

So the computational path between distant positions becomes long.

We can think of it roughly as:

```text
distance between positions ↑
        ↓
dependency path ↑
```

For an RNN, a distant dependency can require roughly:

```text
O(T)
```

sequential steps.

---

# 💡 Why Is a Long Path a Problem?

Every transition transforms the representation.

So long-distance information must survive many transformations.

That can make:

* information preservation harder,
* optimization harder,
* long-range relationships harder to learn,
* gradient flow more difficult.

This is exactly why we previously needed ideas such as:

```text
LSTM
GRU
```

They helped information survive these long recurrent paths.

But they did not remove the paths themselves.

---

# 🌍 Chapter 5 — Didn't LSTM Already Solve This?

This is the obvious question.

We spent a lot of time learning LSTM.

Its cell state allows information to flow more effectively:

```text
c_t
=
f_t ⊙ c_(t-1)
+
i_t ⊙ g_t
```

So surely the problem is solved?

Not quite.

LSTM improves:

# **how well information travels**

But information still travels like this:

```text
c₁
↓
c₂
↓
c₃
↓
...
↓
c₁₀₀
```

To compute:

```text
c₁₀₀
```

we still need:

```text
c₉₉
```

which requires:

```text
c₉₈
```

and so on.

So LSTM solved:

```text
poor information flow
```

but it did not solve:

```text
sequential dependency
```

---

# 💡 Connecting Back to Our Previous Modules

Notice the pattern.

### RNN

```text
Can model sequences
↓
But struggles with long-term dependencies
```

### LSTM / GRU

```text
Improves memory
↓
But still recurrent
```

### Attention

```text
Improves source accessibility
↓
But recurrence still remains
```

We have improved the architecture several times.

But this dependency remains:

```text
h_t
depends on
h_(t-1)
```

---

# 🌍 Chapter 6 — Why Sequential Dependency Became a Bigger Problem

Imagine training on a sentence with:

```text
1000 tokens
```

All 1000 input tokens are already available.

But an RNN cannot simply calculate:

```text
h₁
h₂
h₃
...
h₁₀₀₀
```

independently.

Why?

Because:

```text
h₂ requires h₁

h₃ requires h₂

h₄ requires h₃
```

So the computation is inherently:

```text
Step 1
↓
Step 2
↓
Step 3
↓
...
```

This matters enormously when training large neural networks.

---

# 🌍 Chapter 7 — Think About the GPU

Modern GPUs are incredibly powerful.

But they are especially powerful when we give them:

```text
many calculations
that can happen together
```

For example:

```text
Large Matrix × Large Matrix
```

can contain millions of multiplications.

A GPU can execute many of them simultaneously.

Now compare that with an RNN:

```text
Calculate h₁

wait

Calculate h₂

wait

Calculate h₃

wait
```

The problem isn't simply:

> RNNs perform calculations.

Every model performs calculations.

The problem is:

> **later calculations depend on earlier calculations.**

So the GPU cannot freely parallelize computation across sequence positions.

---

# 🧠 The Parallelism Problem

RNN:

```text
Position 1
    ↓
Position 2
    ↓
Position 3
    ↓
Position 4
```

What we would prefer is something more like:

```text
Position 1    Position 2    Position 3    Position 4
    ↓             ↓             ↓             ↓
       process many positions together
```

But how can we do that?

If we remove recurrence:

```text
h_t ← h_(t-1)
```

how will words communicate with one another?

This sounds impossible.

Until we remember something we already learned.

---

# 🌍 Chapter 8 — Attention Gave Us a Clue

Let's go back to classical Attention.

Suppose the decoder needs information from source position 3.

Attention does not say:

```text
Decoder
↓
Source 10
↓
Source 9
↓
Source 8
↓
...
↓
Source 3
```

Instead, it can directly calculate relevance between:

```text
Decoder State
        ↕
Encoder State h₃
```

Conceptually:

```text
Decoder
────────────────→ h₃
```

Attention created a **direct connection**.

This is the crucial clue.

---

# 🤔 The Big Question

Researchers could now ask:

> **If a decoder state can directly attend to any encoder state...**

why can't:

> **one source position directly attend to another source position?**

Instead of:

```text
animal
 ↓
didn't
 ↓
cross
 ↓
the
 ↓
street
 ↓
because
 ↓
it
```

what if we allowed:

```text
animal
────────────────────→ it
```

directly?

Now we have something fundamentally different.

---

# 🌍 Chapter 9 — The Thought Experiment

Consider again:

> **The animal didn't cross the street because it was tired.**

Suppose every word could look at every other word.

For `it`:

```text
it
↓
look at "The"
look at "animal"
look at "didn't"
look at "cross"
look at "street"
look at "because"
look at "was"
look at "tired"
```

Then `it` could ask:

> Which of these words contains useful information for understanding me?

Perhaps it gives high importance to:

```text
animal
```

and lower importance to unrelated words.

Conceptually:

```text
it
     ↓
compare with all words
     ↓
find relevant words
     ↓
collect information from them
```

Wait...

Doesn't this sound familiar?

It is exactly the idea we learned in Attention.

Except previously:

```text
Decoder
attended to
Encoder states
```

Now:

```text
Token
attends to
tokens in the same sequence
```

This is the key idea behind:

# **Self-Attention**

---

# 🧠 Why "Self"?

Because the attention happens:

```text
within the same sequence
```

Classical encoder-decoder attention:

```text
Decoder Sequence
        ↓
attends to
        ↓
Encoder Sequence
```

Self-Attention:

```text
Sequence
   ↓
attends to itself
   ↓
Same Sequence
```

For example:

```text
animal ────────────────→ it

street ────────────────→ cross

tired ─────────────────→ it
```

Different positions can directly exchange information.

---

# 🌍 Chapter 10 — Something Important Just Happened

Previously, `animal` had to communicate with `it` through several recurrent states.

```text
animal
→
...
→
it
```

Now:

```text
animal
────────→
it
```

The dependency path has become dramatically shorter.

For distant positions, an RNN might need:

```text
O(T)
```

sequential transitions.

A self-attention layer can potentially create a direct interaction in:

```text
one layer
```

This is a fundamental change.

---

# 💡 First Major Benefit — Shorter Dependency Paths

RNN:

```text
A → B → C → D → E
```

For A to affect E:

```text
4 intermediate transitions
```

Self-Attention:

```text
A ─────────────→ E
```

Direct interaction.

This makes long-range relationships much easier to access.

---

# 🌍 Chapter 11 — The Second Benefit Appears

Now notice something else.

In an RNN:

```text
h₂
```

cannot exist before:

```text
h₁
```

because:

```text
h₂ = f(x₂, h₁)
```

But with self-attention, suppose every token starts with its own representation.

```text
x₁
x₂
x₃
x₄
```

All of these already exist.

Each one can independently ask:

```text
Which other tokens matter to me?
```

So the model can calculate interactions for many positions together.

Conceptually:

```text
x₁ ─┐
x₂ ─┼──→ Attention
x₃ ─┼──→ many positions processed together
x₄ ─┘
```

Now sequence positions no longer have to wait for:

```text
previous hidden state
```

before participating.

---

# ⭐ The Parallelism Breakthrough

The old architecture required:

```text
1
↓
2
↓
3
↓
4
```

The new idea allows something closer to:

```text
1     2     3     4
↓     ↓     ↓     ↓
Attention interactions
processed together
```

And those interactions can be implemented largely through:

```text
matrix multiplication
```

which GPUs handle extremely well.

Now we have solved a major architectural limitation of RNNs.

---

# 🌍 Chapter 12 — So Why Not Remove Recurrence?

At this point, the idea becomes almost unavoidable.

If Attention can:

* connect distant positions directly,
* move information between tokens,
* create contextual representations,
* and allow much more parallel processing,

then ask:

> **Why do we still need an RNN to carry information token by token?**

Instead of:

```text
RNN
+
Attention
```

what if we build the sequence model around:

```text
Attention
```

itself?

This was the revolutionary idea.

---

# 🚀 The Transformer

The Transformer removed recurrence from the core sequence-processing architecture.

Instead of:

```text
Token
↓
Recurrent State
↓
Next Recurrent State
↓
Next Recurrent State
```

the model could use:

```text
Token Representations
↓
Attention-Based Interaction
↓
New Contextual Representations
```

The architecture was no longer centered around:

```text
passing a hidden state forward
```

It was centered around:

```text
letting positions directly retrieve
relevant information from other positions
```

---

# 🧠 The Fundamental Shift

RNN philosophy:

> **Carry information through the sequence.**

Transformer philosophy:

> **Directly retrieve relevant information from the sequence.**

This is one of the most useful mental models for understanding the architectural transition.

---

# 🌍 Chapter 13 — But We Just Created a New Problem

This seems perfect.

Remove recurrence.

Let all words interact directly.

Process them in parallel.

Done?

Not quite.

Recall something recurrence gave us automatically.

Consider:

```text
Dog bites man
```

and:

```text
Man bites dog
```

They contain exactly the same words.

But their meaning is completely different because:

# **order matters**

An RNN processes:

```text
token 1
then token 2
then token 3
```

So order is naturally part of its computation.

But if we remove recurrence and process token representations together:

```text
Dog     bites     man
```

how does the model know:

```text
Dog is position 1

bites is position 2

man is position 3
```

Attention alone does not automatically provide that information.

---

# 💡 A New Problem Appears

By removing recurrence, we gain:

```text
parallelism
+
short dependency paths
```

but lose:

```text
built-in sequential ordering
```

Therefore the architecture needs another mechanism to represent:

# **Position**

This will eventually lead to:

# **Positional Encoding**

Notice the pattern.

We solved one limitation.

That immediately exposed another.

Exactly like our earlier modules.

---

# 🌍 Chapter 14 — And Another Question Remains

We have said:

> every token can look at every other token.

But how?

Take the word:

```text
it
```

How does it decide that:

```text
animal
```

is more relevant than:

```text
street
```

How does one token express:

> **What am I looking for?**

And how does another token express:

> **What information do I contain?**

We don't know yet.

We have only discovered the architectural idea.

That missing mechanism will eventually lead us to:

```text
Query
Key
Value
```

But introducing those now would be too early.

First we need to properly understand:

# **Self-Attention**

---

# 🧠 Chapter 15 — The Engineering Progression

Notice how we arrived here.

We did **not** start by saying:

> “Let's invent Transformers.”

Instead:

```text
RNN
        ↓
Can model sequence
        ↓
But long-term information is difficult
        ↓
LSTM / GRU
        ↓
Better memory
        ↓
But Seq2Seq compresses source into one context
        ↓
Attention
        ↓
Dynamic source access
        ↓
But encoder/decoder are still recurrent
        ↓
Why should tokens communicate sequentially?
        ↓
Attention already proves direct retrieval works
        ↓
Let tokens directly attend to other tokens
        ↓
Self-Attention
        ↓
Recurrence may no longer be necessary
        ↓
Transformer
```

This is the Transformer story.

---

# ⭐ Chapter 16 — What Transformer Actually Improved

The Transformer architecture gave us several major advantages.

## 1. Better Training Parallelism

RNN:

```text
token 1
↓
token 2
↓
token 3
```

Transformer-style processing:

```text
token 1    token 2    token 3
   ↓          ↓          ↓
process many positions together
```

---

## 2. Shorter Long-Range Paths

RNN:

```text
1 → 2 → 3 → ... → 100
```

Attention-based interaction:

```text
1 ─────────────────→ 100
```

---

## 3. Better Fit for GPUs / TPUs

Large attention calculations can be expressed using:

```text
matrix operations
```

which modern accelerators execute efficiently.

---

## 4. Easier Scaling

Greater training parallelism made it practical to train:

```text
more data
+
larger models
+
more layers
```

This later became extremely important for large-scale language models.

---

# ⚠️ Chapter 17 — But Transformers Are Not Magic

We should not repeat the mistake of thinking:

```text
Transformer
↓
Everything solved
```

Removing recurrence creates new trade-offs.

Suppose there are:

```text
T
```

tokens.

If every token interacts with every token, we have approximately:

```text
T × T
```

relationships.

So full self-attention has roughly:

```text
O(T²)
```

pairwise interactions.

For:

```text
T = 1000
```

that means:

```text
1000 × 1000
=
1,000,000
```

token-pair score relationships.

So Transformers trade:

```text
sequential recurrence
```

for:

```text
many parallel pairwise interactions
```

---

# 🧠 Architecture Is Always About Trade-Offs

RNN:

```text
Sequential
+
long dependency paths
+
limited sequence parallelism
```

Transformer:

```text
parallelizable
+
short dependency paths
+
many pairwise interactions
```

The Transformer did not eliminate computational cost.

It changed the computation into a form that:

> maps much better to modern parallel hardware.

---

# ⚠️ Chapter 18 — Training Parallelism Does Not Mean Parallel Generation

Another important distinction.

During training, the complete sequence is already known.

So many positions can be processed simultaneously.

But during autoregressive generation:

```text
Generate token 1
↓
Generate token 2
↓
Generate token 3
```

future tokens do not exist yet.

Therefore models like GPT still generate:

> token by token.

So:

```text
Transformer
≠
all computation is always parallel
```

The big breakthrough was mainly:

> **parallel sequence processing during training and within available context.**

---

# 🧠 Engineer's Insight

Look at the pattern that has followed us through this bootcamp.

```text
Perceptron
↓
Cannot model nonlinear boundaries
↓
Hidden Layers

Hidden Layers
↓
Still linear without nonlinear activation
↓
Activation Functions

RNN
↓
Long-term memory problems
↓
LSTM / GRU

Seq2Seq
↓
Fixed Context Bottleneck
↓
Attention

Attention
↓
Dynamic source access
but recurrence remains
↓
Self-Attention / Transformer
```

This is the mindset we want.

Architectures are not random collections of components.

Each new idea exists because:

> **something in the previous architecture was not good enough.**

---

# 🧠 First-Principles Chain

```text
What did Attention solve?
        │
        ▼
The fixed-context bottleneck.
        │
        ▼
What still remained?
        │
        ▼
The encoder and decoder were recurrent.
        │
        ▼
Why is recurrence a problem?
        │
        ▼
Each state depends on the previous state.
        │
        ▼
What does that cause?
        │
        ├───────────────┐
        ▼               ▼
Sequential         Long dependency
computation            paths
        │               │
        └───────┬───────┘
                ▼
Can we avoid passing information
one position at a time?
                │
                ▼
What did Attention already teach us?
                │
                ▼
Representations can retrieve
information directly.
                │
                ▼
Can tokens retrieve information
from other tokens directly?
                │
                ▼
Self-Attention
                │
                ▼
Do we still need recurrence
as the core sequence mechanism?
                │
                ▼
No.
                │
                ▼
Transformer
```

---

# 🎯 Key Takeaways

* Classical Attention solved the fixed-context bottleneck.
* But classical Attention was still attached to recurrent encoders and decoders.
* RNN, LSTM, and GRU all require sequential state dependencies.
* LSTM improves memory but does not remove recurrence.
* Long-distance information in RNNs travels through long computational paths.
* Recurrent dependencies limit parallel processing across sequence positions.
* Classical Attention proved that direct relevance-based retrieval between representations works.
* This inspired the idea of allowing tokens in the same sequence to interact directly.
* That idea is called **Self-Attention**.
* Self-Attention creates shorter dependency paths and much greater training parallelism.
* Removing recurrence creates a new problem: token order is no longer inherently represented.
* Full self-attention also introduces roughly `O(T²)` pairwise interactions.
* The Transformer was therefore not a magical solution; it was a different and highly useful architectural trade-off.

---

# 🔜 Looking Ahead

We have now reached the key idea:

> **Instead of passing information token by token, allow every token to directly retrieve information from other tokens.**

But we have only described the idea.

We still don't know **how** it works.

Take:

> **The animal didn't cross the street because it was tired.**

For the word:

```text
it
```

how does the model decide that:

```text
animal
```

is important?

Why should:

```text
animal
```

receive more attention than:

```text
street
```

How can each token inspect the rest of the sentence and dynamically build a better representation of itself?

That is exactly what we will discover in the next lecture:

# 🚀 **Self-Attention**
