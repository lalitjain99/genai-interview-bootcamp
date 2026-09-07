# 📘 Why Transformers Were Invented

## 🌟 Introduction

We have reached one of the most important transitions in modern AI.

So far, our sequence-modeling journey has been:

```text
RNN
↓
LSTM / GRU
↓
Encoder–Decoder
↓
Seq2Seq
↓
Attention
```

Each architecture solved a problem created by the previous one.

RNNs gave neural networks memory.

LSTMs and GRUs improved long-term information flow.

Encoder–Decoder architectures allowed one sequence to be transformed into another.

Attention solved the fixed-context bottleneck by allowing the decoder to dynamically retrieve information from encoder states.

But at the end of our Attention module, one major problem remained:

> **The architecture was still recurrent.**

The system still looked approximately like:

```text
RNN / LSTM Encoder
+
Attention
+
RNN / LSTM Decoder
```

So researchers asked a much more radical question:

> **If attention is already capable of directly connecting relevant representations, do we really need recurrence at all?**

That question eventually led to:

# **The Transformer**

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to explain:

* why RNN-based architectures were difficult to scale;
* why LSTM and GRU did not solve the parallelism problem;
* why classical attention was not enough by itself;
* what sequential dependency means computationally;
* why long dependency paths matter;
* why GPUs favor parallel matrix operations;
* how attention suggested a way to remove recurrence;
* the difference between cross-attention and self-attention;
* why removing recurrence creates a positional-information problem;
* what the Transformer fundamentally changed;
* what Transformers improved;
* what Transformers did **not** magically solve;
* why the Transformer architecture was such a major turning point.

---

# 📖 Part 1 — Start from the RNN

Recall the basic RNN:

```text
h_t
=
f(x_t, h_(t-1))
```

The current hidden state depends on:

```text
current input
+
previous hidden state
```

So for a sequence:

```text
x_1, x_2, x_3, x_4
```

we compute:

```text
x_1
↓
h_1

h_1 + x_2
↓
h_2

h_2 + x_3
↓
h_3

h_3 + x_4
↓
h_4
```

The important structural fact is:

```text
h_4
cannot be computed
before h_3
```

and:

```text
h_3
cannot be computed
before h_2
```

This creates a:

# **Sequential Dependency Chain**

---

# 🧠 Part 2 — Why Sequential Dependency Matters

Suppose we have:

```text
1000 tokens
```

An RNN conceptually processes:

```text
token 1
↓
token 2
↓
token 3
↓
...
↓
token 1000
```

Even if all 1000 input tokens are already available at training time, the recurrent states cannot generally be computed independently.

That means the architecture cannot fully exploit:

> parallel computation across sequence positions.

---

# ⭐ Part 3 — The Hardware Problem

Modern GPUs and TPUs are extremely good at operations like:

```text
Large Matrix
×
Large Matrix
```

because thousands of numerical operations can happen in parallel.

They prefer computation shaped like:

```text
many operations
at the same time
```

rather than:

```text
operation 1
↓
wait
↓
operation 2
↓
wait
↓
operation 3
```

RNN recurrence therefore creates an important mismatch:

```text
GPU
→ wants parallel work

RNN
→ imposes sequential dependencies
```

---

# 📖 Part 4 — Why Didn't LSTM Solve This?

LSTM fixed an important RNN weakness:

> poor long-term information flow.

Its cell state:

```text
c_t
=
f_t ⊙ c_(t-1)
+
i_t ⊙ g_t
```

creates a better path for information and gradients.

But computationally:

```text
c_t
```

still depends on:

```text
c_(t-1)
```

and:

```text
h_t
```

still depends on previous recurrent state.

So the sequence is still:

```text
step 1
→
step 2
→
step 3
→
...
```

LSTM improved:

# **memory**

but did not solve:

# **sequential computation**

---

# 🧠 Part 5 — Same Problem with GRU

GRU simplified the gating structure.

But:

```text
h_t
```

still depends on:

```text
h_(t-1)
```

So GRU also remains recurrent.

Therefore:

```text
RNN
LSTM
GRU
```

all share one fundamental architectural property:

> **sequence positions are processed through recurrent state dependencies.**

---

# 📖 Part 6 — Bidirectional RNNs Did Not Solve It Either

BiRNN gives:

```text
forward recurrence
+
backward recurrence
```

This improves context because each position can incorporate:

```text
left context
+
right context
```

But both directions are still recurrent:

```text
→h_1 → →h_2 → →h_3
```

and:

```text
←h_3 → ←h_2 → ←h_1
```

So:

```text
Bidirectional
≠
Non-recurrent
```

---

# ⭐ Part 7 — Then Attention Arrived

Classical Seq2Seq compressed the source into:

```text
one context vector
```

Attention instead kept:

```text
h_1, h_2, ..., h_T
```

and allowed a decoder state to retrieve relevant information:

```text
c_t
=
Σ_i α_(t,i) h_i
```

This was a major breakthrough.

The decoder no longer needed to rely on:

```text
one fixed summary
```

---

# 🧠 Part 8 — What Attention Demonstrated

Attention demonstrated something profound:

> **A representation can directly inspect other representations and dynamically decide which ones are relevant.**

For example:

```text
decoder state
↓
compare with h_1
compare with h_2
compare with h_3
...
↓
weighted combination
```

This provided a new way to move information through a neural network.

Not through:

```text
sequential recurrence
```

but through:

```text
direct relevance-based interaction
```

---

# 📖 Part 9 — But Classical Attention Was Still Attached to an RNN

The architecture was still:

```text
Source
↓
RNN Encoder
↓
h_1 ... h_T

             ↑
          Attention
             ↑

Target
↓
RNN Decoder
```

Attention improved:

```text
decoder ↔ encoder communication
```

But recurrence still handled:

```text
source sequence modeling
```

and:

```text
target sequence modeling
```

---

# ⭐ Part 10 — The Critical Question

This naturally creates the question:

> **Why should source tokens communicate through recurrence if attention can connect representations directly?**

Suppose we want token 10 to learn something from token 2.

RNN-style:

```text
2
→
3
→
4
→
5
→
...
→
10
```

What if instead we allowed:

```text
2
────────→
10
```

directly?

That is the conceptual leap toward:

# **Self-Attention**

---

# 🧠 Part 11 — From Decoder-to-Encoder Attention to Position-to-Position Attention

Classical attention:

```text
Decoder State
↓ attends to
Encoder States
```

This is roughly:

```text
target representation
→
source representations
```

The new idea:

```text
Source Position
↓ attends to
Other Source Positions
```

Now:

```text
token
→
token
```

interactions can happen directly within the same sequence.

This is:

# **Self-Attention**

---

# 📖 Part 12 — Why Is It Called "Self" Attention?

Because the:

```text
querying representation
```

and the:

```text
representations being attended to
```

come from the same sequence.

Conceptually:

```text
Sequence
↓
positions attend to positions
within that same sequence
```

For example:

```text
The animal didn't cross the street because it was tired.
```

When constructing the representation of:

```text
it
```

the model may need information from:

```text
animal
```

Instead of passing that information token by token through recurrence, self-attention can create a more direct interaction.

---

# ⭐ Part 13 — Direct Long-Range Interaction

Suppose:

```text
token 1
```

needs information from:

```text
token 100
```

### RNN

Conceptual dependency:

```text
1 → 2 → 3 → ... → 100
```

Path length:

```text
O(T)
```

### Self-Attention

Conceptually:

```text
1 ─────────────→ 100
```

A direct relationship can be established within one attention layer.

This dramatically shortens:

# **dependency paths**

---

# 🧠 Part 14 — Why Shorter Dependency Paths Matter

Long paths can make:

* information propagation harder;
* optimization more difficult;
* long-range relationships harder to learn.

A direct attention connection allows distant positions to exchange information without traversing every intermediate position.

This is especially valuable for relationships like:

```text
pronoun ↔ noun

subject ↔ verb

question ↔ answer evidence

entity ↔ later reference
```

---

# 📖 Part 15 — The Parallelism Breakthrough

Now imagine all source tokens are already available:

```text
x_1
x_2
x_3
...
x_T
```

Instead of:

```text
compute h_1
then h_2
then h_3
```

we could let every position perform attention interactions in the same layer.

Conceptually:

```text
position 1 ─┬─ position 2
            ├─ position 3
            └─ position 4

position 2 ─┬─ position 1
            ├─ position 3
            └─ position 4

...
```

Many of these operations can be expressed using:

> large matrix multiplications.

And that is exactly what modern accelerators are good at.

---

# ⭐ Part 16 — The Fundamental Transformer Trade-Off

RNN:

```text
less all-pairs interaction
+
sequential computation
```

Self-attention:

```text
many pairwise interactions
+
high parallelism across positions
```

So Transformers do not simply make computation disappear.

They exchange one bottleneck:

```text
sequential recurrence
```

for another:

```text
pairwise attention computation
```

---

# 📐 Part 17 — Self-Attention Pairwise Structure

For a sequence length:

```text
T
```

each position may compare against:

```text
T
```

positions.

So roughly:

```text
T × T
```

relationships are created.

Conceptually:

```text
O(T²)
```

attention interactions.

This will later become an important limitation of Transformers.

---

# 🧠 Part 18 — Why Was This Still Attractive?

Because modern accelerators are good at:

```text
parallel matrix operations
```

Even though self-attention performs many comparisons, those comparisons can be highly vectorized.

So architecture design was no longer only about:

> number of arithmetic operations.

It was also about:

> how efficiently hardware can execute those operations.

---

# 📖 Part 19 — A Simple Hardware Analogy

Imagine two jobs.

### Job A

1000 tasks, but each task waits for the previous one:

```text
1
→
2
→
3
→
...
→
1000
```

### Job B

1,000,000 small tasks, but many can happen simultaneously.

With massively parallel hardware, Job B can sometimes be more attractive despite having more total interactions.

That is part of the intuition behind Transformer efficiency.

---

# ⭐ Part 20 — What Did the Transformer Remove?

The Transformer removed recurrence as the primary sequence-processing mechanism.

Instead of:

```text
h_t
=
f(h_(t-1), x_t)
```

the architecture relies on blocks involving:

```text
Attention
+
Feed-Forward Network
+
Residual Connections
+
Normalization
```

Sequence positions are contextualized through:

> attention-based interactions.

---

# 📖 Part 21 — The Famous Architectural Shift

Old sequence modeling:

```text
Recurrence
is the core

Attention
is an add-on
```

Transformer:

```text
Attention
becomes the core interaction mechanism
```

This is the fundamental conceptual change.

---

# 🧠 Part 22 — But Removing Recurrence Creates a Problem

RNNs naturally process:

```text
token 1
then
token 2
then
token 3
```

Therefore sequence order is built into the computation.

But suppose self-attention receives:

```text
I love AI
```

and:

```text
AI love I
```

If the model only sees token representations and pairwise similarity:

> how does it know which token comes first?

---

# ⭐ Part 23 — Attention Alone Does Not Encode Order Automatically

A plain attention operation does not inherently know:

```text
position 1
position 2
position 3
```

Removing recurrence removes the natural ordering mechanism.

Therefore the Transformer needs an explicit way to represent:

# **Position**

This will later lead to:

# **Positional Encoding**

---

# 📖 Part 24 — Another Question: How Do Tokens Decide Relevance?

Suppose the word:

```text
bank
```

appears in:

```text
I deposited money in the bank.
```

The representation needs information from:

```text
money
deposited
```

How should `bank` determine what other words are relevant?

We need representations that express different roles such as:

```text
What am I looking for?

What information do I contain?

What information should I contribute?
```

These questions eventually lead to:

```text
Query
Key
Value
```

But we will not jump there yet.

---

# 🧠 Part 25 — Why We Should Learn Self-Attention Before Q/K/V

The most important concept first is:

> **A sequence position can dynamically retrieve information from other positions in the same sequence.**

Only after that intuition is clear should we ask:

> How is the compatibility actually parameterized?

That leads naturally to Query, Key, and Value.

So our progression will be:

```text
Why Transformer?
↓
Self-Attention
↓
Query / Key / Value
↓
Scaled Dot-Product Attention
```

---

# ⭐ Part 26 — Transformer Architecture at a High Level

Without diving into details yet, a Transformer layer broadly does:

```text
Input Token Representations
↓
Attention
↓
Residual + Normalization
↓
Feed-Forward Network
↓
Residual + Normalization
↓
Contextualized Representations
```

Multiple such blocks can be stacked.

---

# 📖 Part 27 — What Does Attention Do?

Attention primarily enables:

```text
token-to-token interaction
```

---

# 📖 Part 28 — What Does the Feed-Forward Network Do?

After tokens exchange information through attention, each position passes through a neural network.

Conceptually:

```text
Attention
→ gather relevant information

FFN
→ transform/process that information
```

We will study this carefully later.

---

# 📖 Part 29 — What Do Residual Connections Do?

We already learned residual connections in Deep Learning.

Recall:

```text
y
=
F(x) + x
```

Transformers heavily reuse this idea.

Residual connections help:

* preserve information;
* stabilize training;
* train deeper networks.

Our earlier Deep Learning module now connects directly to Transformers.

---

# 🧠 Part 30 — What Does Layer Normalization Do?

Deep architectures need stable activations.

Transformers use:

# **Layer Normalization**

around major sublayers.

Later we will study exactly:

* what it normalizes;
* why BatchNorm is less natural here;
* Pre-LN vs Post-LN.

---

# ⭐ Part 31 — Why Multiple Attention Heads?

One relationship might be:

```text
pronoun → noun
```

Another:

```text
verb → subject
```

Another:

```text
word → nearby modifier
```

Using one attention mechanism may force all relationships into one interaction space.

A natural idea is:

> perform several attention operations in parallel.

This leads to:

# **Multi-Head Attention**

---

# 📖 Part 32 — Why Feed-Forward Layers Are Still Needed

Attention primarily mixes information:

```text
between positions
```

But we also need rich nonlinear transformation:

```text
within each position's representation
```

That is provided by the feed-forward network.

This gives a useful early mental model:

```text
Attention
→ communication

FFN
→ computation
```

This is simplified, but very useful.

---

# 🧠 Part 33 — Transformer Encoder

At a high level:

```text
Input
↓
Token Embeddings
+
Position Information
↓
Transformer Encoder Layers
↓
Contextualized Token Representations
```

Each source position can incorporate information from other source positions.

---

# 📖 Part 34 — Transformer Decoder

The decoder must generate:

```text
token by token
```

and must not look at future target tokens.

So it needs:

```text
masked / causal attention
```

Conceptually:

```text
token t
can see
tokens < t

but not
future tokens
```

This will later become:

# **Causal Attention**

---

# ⭐ Part 35 — Transformer Does Not Mean All Generation Is Parallel

This is a critical misconception.

During training, target tokens are known, so masked attention over many target positions can be computed in parallel.

During autoregressive inference:

```text
generate token 1
↓
generate token 2
↓
generate token 3
```

still happens sequentially.

So Transformers dramatically improve:

> training parallelism

but autoregressive generation can still be sequential.

---

# 🧠 Part 36 — Training vs Inference

### Transformer Training

Many token positions:

```text
can be processed together
```

with causal masking where necessary.

### Autoregressive Inference

Future tokens do not exist yet.

Therefore:

```text
token 1
→ token 2
→ token 3
```

remains sequential.

This distinction will become extremely important when we discuss:

```text
KV Cache
```

later in LLM Internals.

---

# 📖 Part 37 — Transformer Encoder vs Decoder Roles

Transformer architecture introduced both:

```text
Encoder
```

and:

```text
Decoder
```

stacks.

Later models use these pieces differently.

For example:

```text
BERT
→ encoder-focused

GPT
→ decoder-only

T5
→ encoder-decoder
```

But we will not jump into model families until we understand the core Transformer block.

---

# ⭐ Part 38 — Why Transformers Scaled Better

Several factors worked together:

```text
No recurrent sequence dependency
+
Parallel token processing during training
+
Large matrix operations
+
Shorter dependency paths
+
Stackable architecture
```

These properties made Transformers highly suitable for:

* larger datasets;
* larger models;
* accelerator hardware;
* large-scale pretraining.

---

# 🧠 Part 39 — Scaling Became a Major Advantage

Once sequence processing became highly parallelizable, researchers could train:

```text
larger models
+
more layers
+
more data
```

more effectively.

This eventually enabled the scaling trend leading toward:

```text
large pretrained language models
```

But Transformers and LLMs are not synonymous.

A Transformer is:

> an architecture.

An LLM is:

> a large language model, commonly but not necessarily built from Transformer architecture.

---

# 🚨 Part 40 — Transformer ≠ LLM

Do not say:

> "Transformer means LLM."

Transformers are used for:

* language
* vision
* audio
* multimodal systems
* biological sequences
* time-series applications

And small Transformer models are also possible.

So:

```text
Transformer
=
architecture

LLM
=
model category / scale / language objective
```

---

# 📖 Part 41 — Why Transformers Were a Turning Point

Before Transformers, neural sequence modeling was dominated by:

```text
recurrent processing
```

The Transformer demonstrated that:

> recurrence was not necessary for high-quality sequence transduction.

Attention-based layers could perform the core sequence modeling.

That changed the direction of NLP research dramatically.

---

# ⭐ Part 42 — The Core Design Principle

The Transformer asks:

> Instead of carrying information forward one timestep at a time, can every token directly gather information from relevant tokens?

RNN mental model:

```text
Carry information through sequence
```

Transformer mental model:

```text
Retrieve relevant information directly
```

---

# 🧠 Part 43 — Another Useful Comparison

## RNN

```text
information travels
```

through recurrent hidden states.

## Transformer

```text
information is retrieved
```

through attention interactions.

This is simplified but captures an important architectural shift.

---

# 📖 Part 44 — RNN Communication Graph

For:

```text
A B C D
```

a forward RNN gives:

```text
A → B → C → D
```

To connect A and D:

```text
A
→ B
→ C
→ D
```

---

# 📖 Part 45 — Self-Attention Communication Graph

Self-attention can conceptually produce:

```text
A ↔ B
A ↔ C
A ↔ D
B ↔ C
B ↔ D
C ↔ D
```

within one layer.

So distant relationships become:

> directly accessible.

---

# ⭐ Part 46 — But Full Connectivity Costs Something

For four tokens:

```text
4 × 4
=
16
```

attention relationships.

For:

```text
1000 tokens
```

approximately:

```text
1,000 × 1,000
=
1,000,000
```

pairwise score relationships.

This eventually becomes one of the biggest Transformer scaling challenges.

---

# 🧠 Part 47 — Architecture Is Always Trade-Offs

RNN:

```text
Sequential
+
long paths
+
less parallelism
```

Transformer:

```text
Parallelizable
+
short paths
+
many pairwise interactions
```

Neither architecture violates computational reality.

The Transformer simply makes a trade-off much better aligned with modern hardware and large-scale learning.

---

# 📖 Part 48 — What Problems Did Transformers Improve?

Transformers substantially improved:

### 1. Training Parallelism

Tokens can be processed together.

### 2. Long-Range Interaction

Distant positions can interact directly.

### 3. Dependency Path Length

Direct attention creates shorter paths.

### 4. Hardware Utilization

Matrix-heavy computation fits GPUs/TPUs.

### 5. Model Scalability

Architecture scales effectively to large models and datasets.

---

# ⚠️ Part 49 — What Problems Did Transformers NOT Remove?

Transformers still have:

### 1. Attention Complexity

Full self-attention approximately:

```text
O(T²)
```

### 2. Attention Memory

The attention matrix grows:

```text
T × T
```

### 3. Position Information Requirement

Without recurrence, order must be represented explicitly.

### 4. Autoregressive Inference

Decoder-only generation remains sequential.

### 5. Finite Context

Models cannot process unlimited sequences.

---

# ⭐ Part 50 — The Transformer Motivation in One Diagram

```text
RNN
↓
Sequential sequence processing
↓
Long dependency paths
↓
Limited parallelism

LSTM / GRU
↓
Better memory
but still recurrent

Attention
↓
Dynamic source access
but still attached to recurrence

Key Question
↓
Can attention replace recurrence?

Self-Attention
↓
Direct token-to-token interaction

Transformer
↓
Parallel sequence processing
+
short dependency paths
+
attention-based sequence modeling
```

---

# 📖 Part 51 — Our Transformer Learning Path

We will build the Transformer from first principles.

Not by starting with this:

```text
QK^T / √d_k
```

That would be a mistake.

Instead:

```text
Why Transformer?
↓
Why Self-Attention?
↓
What does one token need from another?
↓
Query / Key / Value
↓
Dot Product
↓
Scaling
↓
Softmax
↓
Weighted Values
↓
Multi-Head Attention
↓
Position
↓
FFN
↓
Residual
↓
LayerNorm
↓
Encoder
↓
Decoder
```

Only then will the complete architecture feel natural.

---

# 🧠 Part 52 — The Most Important Concept Before Moving On

Do not memorize:

> “Transformer uses attention instead of RNN.”

Understand **why**.

The progression is:

```text
Recurrence
↓
Good sequence modeling
but poor sequence parallelism

Attention
↓
Proves direct representation interaction works

Self-Attention
↓
Use direct interaction inside a sequence

Transformer
↓
Build the architecture around that interaction
```

---

# 🎤 30-Second Interview Answer

> **Transformers were introduced to overcome important limitations of recurrent sequence models. RNNs, LSTMs, and GRUs process sequence states sequentially, which limits training parallelism and creates long dependency paths. Classical attention solved the fixed-context bottleneck but was still attached to recurrent encoders and decoders. The Transformer made attention the core sequence-modeling mechanism, allowing positions to interact directly, shortening long-range dependency paths, and enabling much greater parallelism on modern accelerators.**

---

# 🎤 Why Didn't LSTM Solve the Need for Transformers?

> **LSTM improved memory and gradient flow but remained recurrent. Each hidden state still depends on the previous state, so sequence computation remains inherently sequential. Transformers address that architectural bottleneck by removing recurrence from the core sequence-processing mechanism.**

---

# 🎤 Why Was Attention the Key Clue?

> **Classical attention showed that a model could dynamically retrieve relevant information by directly comparing representations. Transformers generalized this idea from decoder-to-encoder retrieval to direct position-to-position interaction within a sequence through self-attention.**

---

# 🎤 Did Transformers Make Everything Parallel?

> **No. Transformer training allows much greater parallelism across sequence positions because recurrence is removed, but autoregressive inference still generates tokens sequentially. Full self-attention also introduces quadratic compute and memory growth with sequence length.**

---

# 🚨 High-Yield Traps

### ❌ Transformers were invented because attention did not work

No.

Attention worked extremely well.

Transformers expanded its role.

---

### ❌ LSTM solved the sequential-computation problem

No.

LSTM is still recurrent.

---

### ❌ Transformer means no sequential inference

No.

Autoregressive generation remains sequential.

---

### ❌ Self-attention is computationally free

No.

Full self-attention creates approximately `T²` interactions.

---

### ❌ Transformer automatically knows token order

No.

Removing recurrence creates the need for explicit position information.

---

### ❌ Transformer and LLM mean the same thing

No.

Transformer is an architecture.

---

# 📐 Complexity / Architecture Card

### RNN State

```text
h_t
=
f(x_t, h_(t-1))
```

### Distant RNN Dependency Path

```text
O(T)
```

### Full Self-Attention Pairwise Structure

```text
T × T
```

Conceptually:

```text
O(T²)
```

### Core Architectural Shift

```text
Recurrence
↓
Sequential state propagation
```

becomes:

```text
Self-Attention
↓
Direct position-to-position interaction
```

---

# ⚡ 10 Things to Know Cold

1. RNN/LSTM/GRU sequence computation is recurrent.
2. Recurrence limits sequence-level parallelism.
3. LSTM improves memory but does not remove recurrence.
4. Classical attention solves source accessibility but remains attached to RNNs.
5. Attention demonstrated that direct representation interaction works.
6. Self-attention allows positions within one sequence to interact directly.
7. Direct interaction shortens long-range dependency paths.
8. Removing recurrence improves training parallelism.
9. Full self-attention introduces roughly `O(T²)` pairwise interactions.
10. Transformers change architectural trade-offs; they do not eliminate all computational limitations.

---

# 🧠 Final Mental Model

The entire motivation can be remembered as:

```text
RNN
↓
Good sequence modeling
but sequential

LSTM / GRU
↓
Better memory
but still sequential

Attention
↓
Better source access
but recurrence still remains

Key Insight
↓
Direct representation interaction works

New Question
↓
Why use recurrence at all?

Self-Attention
↓
Positions directly interact

Transformer
↓
Attention becomes the core
sequence-modeling mechanism
```

Shortest version:

# **Attention solved access. Transformer removed recurrence from the core sequence model.**

---

# 🔗 Next Chapter

We now understand **why** Transformers were needed.

But we have not yet explained the mechanism that makes them possible.

The next question is:

> **How can one token dynamically retrieve information from other tokens in the same sequence?**

That takes us to:

# **02_Self_Attention**

And we will derive it from first principles before introducing:

```text
Query
Key
Value
```
