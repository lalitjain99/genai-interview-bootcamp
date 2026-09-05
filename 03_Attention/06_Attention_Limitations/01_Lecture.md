# 📘 Attention Limitations

## 🌟 Introduction

Attention solved one of the biggest weaknesses of classical Seq2Seq models.

Before attention, the encoder had to compress the entire source sequence into a single fixed context vector:

```text
x_1, x_2, ..., x_T
↓
Encoder
↓
one fixed context c
↓
Decoder
```

That created a severe bottleneck.

Attention changed the communication pattern:

```text
Encoder
↓
h_1, h_2, ..., h_T

Decoder step t
↓
compute α_(t,1), ..., α_(t,T)
↓
c_t = Σ_i α_(t,i) h_i
```

Now the decoder could dynamically retrieve source information at each generation step.

This was a major breakthrough.

But it did **not** solve everything.

The architecture was still mostly:

```text
RNN / LSTM Encoder
+
Attention
+
RNN / LSTM Decoder
```

So a deeper question remained:

> **If attention solved the fixed-context bottleneck, what important problems were still left?**

Those remaining problems eventually motivated a much larger architectural shift:

# **Transformers**

This chapter is the bridge.

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to explain:

* what problem attention solved;
* what problems attention did **not** solve;
* why recurrent encoders still limit parallelism;
* why recurrent decoders remain sequential;
* how long dependency paths remain in RNN-based architectures;
* why attention itself introduces pairwise computation;
* why global attention can become expensive;
* why local attention trades accessibility for efficiency;
* why attention still depends on encoder representation quality;
* why classical attention does not remove vanishing/exploding-gradient issues entirely;
* why dynamic source access is different from removing recurrence;
* why these limitations naturally motivate self-attention and Transformers.

---

# 📖 Part 1 — First, Be Precise About What Attention Fixed

Classical Seq2Seq had:

```text
source sequence
↓
encoder
↓
one fixed vector c
↓
decoder
```

The problem was:

> every target step had access to the source only through one compressed representation.

Attention replaced this with:

```text
h_1, h_2, ..., h_T
↓
dynamic retrieval
↓
c_t
```

So attention solved:

# **Source Accessibility**

The decoder no longer had to depend on only:

```text
h_T
```

or one fixed global context.

---

# 🧠 Part 2 — What Attention Did NOT Automatically Solve

Attention did **not** automatically eliminate:

* recurrent source processing;
* recurrent target processing;
* sequential dependencies;
* long computational paths;
* autoregressive decoding;
* global attention cost;
* imperfect encoder representations.

This distinction is extremely important.

A common wrong mental model is:

```text
Attention
↓
RNN problems solved
```

The better model is:

```text
Attention
↓
fixed-context communication problem largely solved

but

RNN recurrence
+
sequential computation
+
other limitations
remain
```

---

# ⭐ Part 3 — Limitation 1: The Encoder Is Still Recurrent

Consider an RNN encoder:

```text
h_1 = f(x_1, h_0)

h_2 = f(x_2, h_1)

h_3 = f(x_3, h_2)

...
```

To calculate:

```text
h_3
```

we need:

```text
h_2
```

To calculate:

```text
h_2
```

we need:

```text
h_1
```

So source states cannot generally be computed independently.

---

# 🧠 Part 4 — Sequential Dependency

The encoder computation chain is:

```text
x_1
↓
h_1
↓
h_2
↓
h_3
↓
...
↓
h_T
```

Even if all input tokens are already known, recurrent dependence means:

> timestep `t+1` must wait for timestep `t`.

This limits:

# **parallel computation across sequence positions**

---

# 🧮 Part 5 — Simple Example

Suppose the source contains:

```text
100 tokens
```

An RNN encoder conceptually computes:

```text
h_1
then
h_2
then
h_3
...
then
h_100
```

We cannot simply compute all 100 recurrent states independently in one step because:

```text
h_t
depends on
h_(t-1)
```

Attention does not change this encoder recurrence.

---

# ⭐ Part 6 — Why Parallelism Matters

Modern hardware such as GPUs and TPUs performs extremely well when many operations can be executed together.

For example:

```text
large matrix operation
```

can often be processed very efficiently.

But recurrence introduces a serial dependency:

```text
step 1
↓
step 2
↓
step 3
↓
step 4
```

So even with attention:

> recurrent source processing can remain a training-time bottleneck.

---

# 📖 Part 7 — Limitation 2: The Decoder Is Still Recurrent

The same issue appears on the target side.

A recurrent decoder computes:

```text
s_t
=
f(y_(t-1), s_(t-1), ...)
```

So:

```text
s_t
```

depends on:

```text
s_(t-1)
```

The decoder states must therefore be produced sequentially.

---

# 🧠 Part 8 — Attention Does Not Remove Decoder Recurrence

Luong, for example:

```text
s_t
↓
attention
↓
c_t
↓
prediction
```

But before attention can use:

```text
s_t
```

the model still has to compute:

```text
s_t
```

from the previous recurrent state.

So attention improves:

> source retrieval

but it does not eliminate:

> decoder state dependency.

---

# ⭐ Part 9 — Training vs Inference Nuance

There are actually two different sequential issues.

## Recurrent computation

Even during training, recurrent states depend on previous states.

```text
s_1 → s_2 → s_3 → ...
```

## Autoregressive generation

During inference:

```text
y_1
↓
y_2 conditioned on y_1
↓
y_3 conditioned on y_1,y_2
```

So generation itself is sequential.

These are related but distinct concepts.

---

# ⚠️ Part 10 — Do Not Confuse Recurrence with Autoregression

A model can be:

* recurrent;
* autoregressive;
* both;
* neither.

Classical RNN decoders are typically both.

Attention itself changes neither property automatically.

This distinction becomes very important once we reach Transformers.

---

# 📖 Part 11 — Limitation 3: Long Computational Dependency Paths

Attention gives the decoder direct access to:

```text
h_i
```

But how was:

```text
h_i
```

created?

In an RNN encoder:

```text
h_i
```

is built through recurrence.

For information from source token 1 to influence:

```text
h_T
```

it travels through many recurrent transitions:

```text
h_1
→
h_2
→
h_3
→
...
→
h_T
```

---

# 🧠 Part 12 — What Is a Dependency Path?

A dependency path means:

> how many computational transformations information must pass through for one position to affect another.

For distant RNN positions:

```text
token 1
→ token 2
→ token 3
→ ...
→ token 100
```

the path can be long.

Longer paths can make:

* information propagation harder;
* optimization more difficult;
* gradient flow more fragile.

---

# ⭐ Part 13 — Attention Helps One Path, But Not All Paths

Attention creates a shortcut from:

```text
decoder step t
```

to:

```text
encoder state h_i
```

That is very valuable.

But:

```text
h_i
```

itself may still have required sequential recurrent computation.

Therefore attention improves:

> encoder-to-decoder communication paths

without fully removing:

> recurrent paths inside the encoder or decoder.

---

# 📖 Part 14 — Limitation 4: Gradient Problems Are Not Magically Gone

We previously learned that RNN training can suffer from:

* vanishing gradients;
* exploding gradients.

LSTM and GRU mitigate these problems.

Attention further helps because the decoder gains more direct access to source states.

But attention does **not** mathematically remove all recurrent gradient paths.

The recurrent encoder and decoder are still trained through time.

So:

> BPTT still exists.

---

# 🧠 Part 15 — Important Interview Distinction

Wrong:

> "Attention solved vanishing gradients."

Better:

> "Attention gives the decoder shorter access paths to encoder states and reduces reliance on a single recurrent summary, but the recurrent encoder-decoder components can still face gradient and sequential-training issues."

---

# 📖 Part 16 — Limitation 5: Attention Cannot Recover Information the Encoder Never Represented

This is one of the most important conceptual limitations.

Attention computes:

```text
c_t
=
Σ_i α_(t,i) h_i
```

Notice the values being combined are:

```text
h_i
```

So attention can only retrieve information that exists in the encoder states.

---

# 🧠 Part 17 — Attention Is Retrieval, Not Magic Reconstruction

Suppose important information was poorly encoded.

Then even if:

```text
α_(t,i) = 0.95
```

attention is strongly selecting:

```text
h_i
```

but if `h_i` itself does not contain enough useful information:

> the decoder still cannot recover what was lost.

Mental model:

```text
Attention
=
better access to stored representations

NOT

creation of missing information
```

---

# ⭐ Part 18 — Connection to Earlier "Accessibility vs Capacity"

We previously separated:

```text
capacity
```

from:

```text
accessibility
```

Attention mainly improves:

# **accessibility**

It lets the decoder retrieve different encoder states dynamically.

But representation quality and capacity still depend on:

* encoder architecture;
* hidden dimensions;
* training;
* data.

This distinction remains crucial.

---

# 📖 Part 19 — Limitation 6: Global Attention Requires Many Comparisons

At each decoder timestep:

```text
t
```

global attention compares the decoder state with:

```text
h_1, h_2, ..., h_(T_src)
```

So approximately:

```text
T_src
```

scores are computed per target timestep.

Across:

```text
T_tgt
```

target steps:

```text
T_src × T_tgt
```

source-target score interactions are required.

---

# 📐 Part 20 — Complexity View

Conceptually:

```text
Global Classical Attention
≈
O(T_src × T_tgt)
```

for the pairwise source-target scoring stage.

If both source and target sequences become long, the number of interactions grows.

---

# 🧮 Part 21 — Numerical Example

Suppose:

```text
T_src = 1,000
T_tgt = 500
```

Then global attention conceptually evaluates:

```text
1,000 × 500
=
500,000
```

source-target pairwise scores.

And that is only the attention-scoring relationship, not the rest of the network.

---

# 🧠 Part 22 — Why This Was Still Acceptable

For many machine-translation tasks of the era, sequence lengths were manageable.

Also, multiplicative attention made these operations efficient.

So attention remained highly practical.

But as sequence lengths increase, the scaling becomes increasingly relevant.

---

# 📖 Part 23 — Limitation 7: Local Attention Trades Cost for Accessibility

Luong proposed Local Attention partly to reduce source-wide search.

Instead of:

```text
attend everywhere
```

we do:

```text
predict relevant region
↓
attend locally
```

This reduces comparisons.

But a new risk appears:

> What if the important information lies outside the selected window?

---

# 🧠 Part 24 — Global vs Local Trade-Off

Global:

```text
all source states available
+
more computation
```

Local:

```text
fewer source states
+
less computation
-
possible missed information
```

So Local Attention does not eliminate the fundamental trade-off.

It changes it.

---

# 📖 Part 25 — Limitation 8: Dynamic Attention Still Happens One Decoder Step at a Time

Consider:

```text
c_1
c_2
c_3
...
```

In a recurrent decoder, the query-like decoder state changes sequentially:

```text
s_1
↓
s_2
↓
s_3
```

Therefore the corresponding attention operations are also driven:

> one decoder timestep at a time.

Even if each source scan is vectorized, decoder-time parallelism remains limited.

---

# 🧠 Part 26 — Source-Wise Parallelism vs Target-Wise Sequentiality

At a single target timestep, we can often compute scores against all source states together:

```text
h_1
h_2
h_3
...
h_T
```

This is good parallelism.

But target steps themselves still follow:

```text
t=1
↓
t=2
↓
t=3
```

because recurrent state generation is sequential.

So classical attention gives:

> parallelism across source comparisons

but not full parallelism across target positions.

---

# ⭐ Part 27 — This Distinction Is Crucial

Do not say:

> "Luong is matrix multiplication, therefore the whole model is parallel."

Wrong.

Only parts of the attention computation can be vectorized.

The surrounding recurrent architecture still imposes sequential dependencies.

---

# 📖 Part 28 — Limitation 9: Encoder Recurrence Can Still Bottleneck Long-Range Representation

Suppose we want source token 100 to have a representation informed by token 1.

In a forward RNN, information must flow through:

```text
1
→
2
→
3
→
...
→
100
```

A BiRNN improves bidirectional context.

But each direction still uses recurrence.

So long-range representation remains tied to:

> sequential information propagation.

---

# 🧠 Part 29 — The Deeper Question

At this point, researchers had achieved:

```text
dynamic source retrieval
```

using attention.

That raises an obvious question:

> If attention can directly connect one representation to another, why should distant source tokens communicate only through recurrent chains?

Instead of:

```text
token 1
→ h_1
→ h_2
→ h_3
→ ...
→ h_100
```

could token 100 somehow access token 1 more directly?

This question points toward:

# **Self-Attention**

---

# 📖 Part 30 — Cross-Attention vs Source-Side Interaction

Classical encoder-decoder attention primarily creates connections like:

```text
decoder state
→ encoder states
```

It improves communication:

```text
target side
↔
source side
```

But inside the encoder, source positions are still related through the recurrent architecture.

A more radical idea would be:

```text
source position
↔
source position
```

directly.

That is the conceptual seed of:

> self-attention.

---

# 🧠 Part 31 — Why "Self" Attention?

Classical attention:

```text
decoder representation
attends to
encoder representations
```

Self-attention:

```text
representation within a sequence
attends to
other representations in that same sequence
```

This can allow direct position-to-position interaction without requiring recurrence between every intermediate position.

We will develop this carefully in the Transformer module.

---

# 📖 Part 32 — Limitation 10: Recurrence Limits Training Throughput

Training large models means processing:

* many sequences;
* many positions;
* many layers;
* many batches.

If sequence positions must be processed sequentially:

```text
1 → 2 → 3 → ... → T
```

hardware cannot exploit all available parallelism across the sequence dimension.

This becomes increasingly important as:

```text
model size ↑
data size ↑
sequence length ↑
```

---

# ⭐ Part 33 — Why This Becomes a Scaling Problem

Imagine two architectures.

### Architecture A

```text
position 2 waits for position 1
position 3 waits for position 2
...
```

### Architecture B

```text
many positions can be processed together
```

Even if individual operations in B are expensive, modern accelerators can make the second architecture attractive because:

> large parallel matrix operations scale well.

This becomes one of the key motivations behind Transformer design.

---

# 📖 Part 34 — Limitation 11: Sequential Computation Makes Very Deep Temporal Paths

Suppose a signal needs to influence something 100 timesteps away.

RNN:

```text
1 → 2 → 3 → ... → 100
```

Path length:

```text
O(T)
```

Direct attention-style interaction:

```text
1 → 100
```

can conceptually have a much shorter dependency path.

Shorter paths can help:

* information flow;
* gradient flow;
* modeling distant relationships.

---

# 🧠 Part 35 — Dependency Path vs Computational Complexity

These are different metrics.

An architecture can have:

* shorter dependency paths;
* but more pairwise computation.

Attention is a good example.

Direct interactions can shorten paths while introducing more comparisons.

So architecture design is always about:

> trade-offs.

---

# 📖 Part 36 — Limitation 12: Pairwise Attention Can Become Memory Intensive

Global attention stores weights or intermediate scores involving many source-target pairs.

For:

```text
T_src × T_tgt
```

relationships, memory grows with the alignment structure.

For classical Seq2Seq lengths this may be manageable.

But the broader lesson is:

> direct pairwise interaction improves accessibility but carries computational and memory costs.

This will become even more important when we study self-attention.

---

# 🧠 Part 37 — Important Foreshadowing

Self-attention will later allow:

```text
every source position
to interact with
every source position
```

That improves parallelism and path length.

But it also creates:

```text
T × T
```

pairwise interactions.

So Transformers do not magically remove all scaling problems.

They change the trade-off.

Keep that in mind.

---

# 📖 Part 38 — Limitation 13: Classical Attention Is Still Architecturally Complex

A recurrent attention model can contain:

```text
Encoder RNN/LSTM
+
Decoder RNN/LSTM
+
Attention Scorer
+
Attention Softmax
+
Context Combination
+
Output Projection
```

There are several interacting stateful components.

This creates complexity around:

* training;
* state management;
* beam search;
* hidden-state reordering;
* inference.

A simpler highly parallel architecture became attractive.

---

# 🧠 Part 39 — Attention Was Initially an Add-On

Historically, attention did not initially replace recurrence.

It was attached to recurrent Seq2Seq:

```text
RNN
+
Attention
```

The revolutionary next question was:

> What if attention is not just an auxiliary component?

What if it becomes:

> the primary mechanism for information interaction?

That is the conceptual leap toward Transformers.

---

# ⭐ Part 40 — Summarize the Remaining Problems

After classical attention, we still have:

```text
1. Recurrent encoder
   → sequential source processing

2. Recurrent decoder
   → sequential target-state processing

3. Autoregressive inference
   → token-by-token generation

4. Long recurrent dependency paths
   → difficult long-range information flow

5. BPTT / recurrent gradient issues
   → not fully removed

6. Encoder representation limitation
   → attention cannot retrieve missing information

7. Global attention
   → T_src × T_tgt comparisons

8. Local attention
   → efficiency/accessibility trade-off

9. Limited sequence-position parallelism
   → hardware utilization bottleneck

10. Stateful architecture complexity
    → training/inference complexity
```

---

# 📖 Part 41 — Which Limitation Directly Motivates Transformers?

There is no single limitation.

Transformers emerge from a combination of pressures:

```text
Need long-range interaction
+
Need shorter dependency paths
+
Need more sequence parallelism
+
Attention already works extremely well
```

Then comes the key idea:

> **Can we build the entire sequence model around attention rather than recurrence?**

---

# 🧠 Part 42 — The Critical Thought Experiment

Suppose we remove recurrence.

Now source token representations no longer automatically know:

```text
what came before
```

or:

```text
what came after
```

So we need a mechanism where tokens can directly exchange information.

Attention can potentially do that.

Imagine:

```text
token 1 ↔ token 2
token 1 ↔ token 3
token 1 ↔ token 4
...
token 3 ↔ token 100
```

Now each position can gather information from others.

That is a radical shift from:

```text
sequential propagation
```

to:

```text
direct interaction
```

---

# ⭐ Part 43 — But We Immediately Create New Questions

If token positions directly attend to each other:

1. How does a token decide which other token is relevant?
2. What representations should be compared?
3. How do we distinguish:

   * what I am looking for;
   * what information another token offers?
4. How do we preserve token order without recurrence?
5. How do we allow multiple kinds of relationships simultaneously?

These questions will lead us to:

* Query
* Key
* Value
* Scaled Dot-Product Attention
* Multi-Head Attention
* Positional Encoding

But not yet.

First, understand why recurrence became the target for removal.

---

# 📖 Part 44 — What Attention Did to the Original Seq2Seq Bottleneck

Evolution so far:

```text
Classic Seq2Seq
↓
one fixed context
↓
source bottleneck

Attention
↓
dynamic source access
↓
fixed-context bottleneck largely removed
```

But:

```text
RNN Encoder
+
RNN Decoder
```

remain.

The next evolution asks:

```text
Can attention replace recurrence itself?
```

---

# 🧠 Part 45 — Source Access vs Sequence Modeling

This distinction captures the whole chapter.

Classical attention primarily solved:

# **Source Access**

But RNNs were still responsible for:

# **Sequence Modeling**

The Transformer idea will be:

> use attention itself as a major sequence-modeling mechanism.

That is the architectural leap.

---

# 📖 Part 46 — A Simple Analogy

Imagine 100 people standing in a line.

### RNN-style communication

Person 1 tells person 2.

Person 2 tells person 3.

Person 3 tells person 4.

Eventually information reaches person 100.

```text
1 → 2 → 3 → ... → 100
```

### Attention-style direct communication

Person 100 can directly ask:

> Which people contain information relevant to me?

and communicate with them directly.

```text
1 ─────────→ 100
27 ────────→ 100
65 ────────→ 100
```

That greatly shortens communication paths.

The next question is:

> Can every position communicate this way?

That is the essence of self-attention.

---

# ⭐ Part 47 — Why Attention Was Such a Strong Clue

Bahdanau and Luong already demonstrated that:

> learned relevance-based retrieval works.

So researchers had evidence that learned weighted interaction between states was powerful.

The remaining step was to generalize that idea beyond:

```text
decoder → encoder
```

toward:

```text
position → position
```

---

# 📐 Part 48 — Classical Architecture vs Emerging Idea

### Classical Attention-Based Seq2Seq

```text
Source
↓
Recurrent Encoder
↓
h_1 ... h_T

Target
↓
Recurrent Decoder
↓
s_t

s_t
↓
Attention over h_i
↓
c_t
```

### Emerging Idea

```text
Source Representations
↓
positions directly interact
↓
contextualized representations
```

No recurrent chain is required for source positions to communicate.

This will become the Transformer encoder.

---

# ⚠️ Part 49 — Transformers Will Not Solve Everything

Do not create another incorrect mental model:

```text
RNN limitations
↓
Transformer
↓
all problems disappear
```

Transformers introduce their own limitations:

* quadratic self-attention cost;
* memory usage;
* positional-information requirements;
* autoregressive decoder inference still sequential in GPT-like models;
* context-window limits.

Good architecture thinking means:

> understanding which trade-offs changed.

---

# ⭐ Part 50 — The Most Important Bridge

The progression is:

```text
RNN
↓
sequence information carried recurrently

Seq2Seq
↓
encoder communicates through one context

Attention
↓
decoder dynamically retrieves encoder states

Remaining problem
↓
encoder/decoder still recurrent

New idea
↓
let positions interact directly with attention

Result
↓
Transformer
```

---

# 🎤 30-Second Interview Answer

> **Classical attention solved the fixed-context bottleneck by letting each decoder step dynamically retrieve encoder states, but it did not remove the recurrent architecture. The encoder and decoder still process states sequentially, which limits parallelism, creates long dependency paths, and retains recurrent training issues. Global attention also requires source-target pairwise comparisons, and attention cannot recover information that the encoder failed to represent. These limitations motivated the idea of using attention itself for sequence modeling, eventually leading to self-attention and Transformers.**

---

# 🎤 What Did Attention Solve vs Not Solve?

> **Attention primarily solved source accessibility: the decoder no longer relied on one fixed encoder summary. It did not automatically solve recurrence, sequential computation, autoregressive decoding, recurrent gradient paths, or pairwise attention cost.**

---

# 🎤 Why Were Transformers Needed If Attention Already Existed?

> **Because classical attention was still layered on top of recurrent encoder-decoder networks. Attention improved communication between encoder and decoder, but recurrence still limited parallelism and created long sequential dependency paths. Transformers made attention the primary sequence-interaction mechanism and removed recurrence from the core architecture.**

---

# 🚨 High-Yield Traps

### ❌ Attention removed all RNN limitations

No.

---

### ❌ Attention solved vanishing gradients completely

No.

---

### ❌ Attention makes the whole model parallel

No.

---

### ❌ Multiplicative attention means all target timesteps run in parallel

No.

---

### ❌ Attention can recover information missing from encoder states

No.

---

### ❌ Local attention gives global access at lower cost

No.

It restricts accessibility.

---

### ❌ Transformer means all sequential computation disappears

No.

Autoregressive generation can still be sequential.

---

# 📐 Complexity Card

### Recurrent encoder dependency

```text
h_t depends on h_(t-1)
```

Sequential path:

```text
O(T)
```

across distant sequence positions.

### Global classical attention interactions

```text
T_src × T_tgt
```

Conceptually:

```text
O(T_src T_tgt)
```

### Alignment matrix

```text
A ∈ R^(T_tgt × T_src)
```

---

# ⚡ 10 Things to Know Cold

1. Attention solved the fixed source-context bottleneck.
2. It did not remove recurrent encoding.
3. It did not remove recurrent decoding.
4. RNN sequence positions remain sequentially dependent.
5. Attention does not fully remove BPTT/gradient issues.
6. Attention cannot retrieve information absent from encoder states.
7. Global attention needs source-target pairwise comparisons.
8. Local attention trades global accessibility for efficiency.
9. Classical attention improves source access, not necessarily source-side sequence modeling.
10. These limitations motivate replacing recurrence with attention-based sequence interaction.

---

# 🧠 Final Mental Model

Start with the success:

```text
Fixed Context
↓
Attention
↓
Dynamic Source Access
```

Then ask what remains:

```text
Dynamic Source Access
+
Recurrent Encoder
+
Recurrent Decoder
↓
Sequential Computation
+
Long Dependency Paths
+
Limited Parallelism
```

That creates the next question:

```text
Can attention do more than retrieve source states?

Can attention itself model relationships
between sequence positions?
```

That leads to:

# **Self-Attention**

and ultimately:

# **Transformers**

---

# 🔗 End of Classical Attention Module

We have now completed the conceptual journey:

```text
01_Attention_Motivation
↓
Why fixed context fails

02_Attention_Mechanism
↓
Generic dynamic source retrieval

03_Bahdanau_Additive_Attention
↓
Nonlinear additive scoring

04_Luong_Multiplicative_Attention
↓
Multiplicative scoring

05_Attention_Alignment_And_Visualization
↓
Understanding learned source-target alignment

06_Attention_Limitations
↓
Why attention + recurrence is still not enough
```

The next major module begins with the question:

> **Can we remove recurrence and build sequence modeling around attention itself?**

# **04 — Transformer Architecture**

And the first chapter should begin with:

# **Why Transformers Were Invented**
