# 💡 Attention Limitations — Assignment

> Goal: Practice distinguishing what classical attention solved from what remained, reason about recurrence and parallelism, analyze complexity and dependency paths, and connect these limitations to the motivation for self-attention and Transformers.

---

# 🧠 1. What Did Attention Actually Solve?

Consider vanilla Seq2Seq:

```text id="e03msx"
Source
↓
Encoder
↓
One Fixed Context Vector
↓
Decoder
```

What core limitation did attention address?

### Answer

Attention addressed the:

> **fixed-context bottleneck**

Instead of forcing every decoder step to rely on one compressed source representation, attention allows dynamic access to:

```text id="8o7ehu"
h_1, h_2, ..., h_T
```

through:

```text id="kwpnks"
c_t
=
Σ_i α_(t,i) h_i
```

So attention primarily improves:

# **Source Accessibility**

---

# 🔍 2. Solved or Still Remaining?

Classify each item as:

```text id="yr4thg"
Solved / Largely Improved
```

or:

```text id="g7e67l"
Still Remaining
```

### A. One fixed source context vector

### B. Dynamic source retrieval

### C. Recurrent encoder computation

### D. Recurrent decoder computation

### E. Autoregressive inference

### F. Long recurrent dependency paths

### G. Source-target alignment

### H. BPTT

### Answer

| Item                     | Status           |
| ------------------------ | ---------------- |
| Fixed source context     | ✅ Largely solved |
| Dynamic source retrieval | ✅ Solved         |
| Recurrent encoder        | ❌ Remains        |
| Recurrent decoder        | ❌ Remains        |
| Autoregressive inference | ❌ Remains        |
| Long recurrent paths     | ❌ Remain         |
| Source-target alignment  | ✅ Improved       |
| BPTT                     | ❌ Remains        |

---

# 📐 3. Why Can an RNN Encoder Not Fully Parallelize Sequence Positions?

Suppose:

```text id="qh793l"
h_t
=
f(x_t, h_(t-1))
```

Can:

```text id="4eeqfc"
h_1, h_2, h_3, h_4
```

all be computed independently at the same time?

### Answer

No.

Because:

```text id="fdhfj2"
h_2 depends on h_1
h_3 depends on h_2
h_4 depends on h_3
```

So there is a sequential dependency chain:

```text id="ykrya5"
h_1
→
h_2
→
h_3
→
h_4
```

This limits parallelism across sequence positions.

---

# 🧠 4. Why Does Attention Not Remove Encoder Recurrence?

A candidate says:

> “Attention directly accesses all encoder states, so recurrence no longer matters.”

What is wrong?

### Answer

Attention can directly access:

```text id="4d9yez"
h_i
```

but each `h_i` may still have been produced through:

```text id="jm8j23"
h_i = f(x_i, h_(i-1))
```

So attention shortens:

> decoder → encoder-state access

but does not eliminate:

> recurrent computation used to create those encoder states.

---

# 🔁 5. Recurrence vs Autoregression

Match the concepts.

### Statement A

```text id="ihhucj"
state_t depends on state_(t-1)
```

### Statement B

```text id="akvwgg"
output_t depends on previous generated outputs
```

### Answer

A:

```text id="cgn7kv"
Recurrence
```

B:

```text id="4xsk9l"
Autoregression
```

A classical RNN decoder is commonly:

```text id="ngw6px"
recurrent
+
autoregressive
```

---

# 🧠 6. Does Attention Remove Autoregressive Inference?

Suppose inference works as:

```text id="16niin"
generate y_1
↓
feed y_1 back
↓
generate y_2
↓
feed y_2 back
```

Does adding attention make all target tokens generated simultaneously?

### Answer

No.

Attention changes how the decoder accesses source information.

It does not automatically change:

```text id="4gr5os"
P(y_t | y_<t, X)
```

So autoregressive generation remains sequential.

---

# 🧮 7. Long Dependency Path

Suppose information at source position 1 must affect position 10 in a forward RNN.

Write the path.

### Answer

Conceptually:

```text id="r98rrb"
1
→
2
→
3
→
4
→
5
→
6
→
7
→
8
→
9
→
10
```

The number of recurrent transitions grows with the distance.

For distant positions, dependency-path length is approximately:

```text id="7wsl21"
O(T)
```

---

# 🧠 8. What Path Does Attention Shorten?

Without attention, a decoder may depend heavily on the final encoder summary.

With attention, the decoder can access:

```text id="xmijkb"
h_2
```

directly.

Which path is improved?

### Answer

Attention shortens:

> **encoder-to-decoder communication**

For example:

```text id="epal3x"
decoder timestep t
──────────────→
h_2
```

But it does not necessarily shorten the recurrent path used to construct:

```text id="5pohrc"
h_2
```

itself.

---

# 🧠 9. Did Attention Solve Vanishing Gradients?

A candidate says:

> “Once attention was introduced, vanishing gradients were solved.”

Is this correct?

### Answer

No.

Attention can provide shorter access paths and reduce reliance on one final recurrent summary.

But recurrent encoder and decoder components still use:

```text id="9fkp0t"
BPTT
```

So recurrent gradient issues can still exist.

A better statement is:

> Attention mitigates some long-path communication problems but does not mathematically eliminate vanishing/exploding gradients in recurrent components.

---

# 🧠 10. Accessibility vs Representation

Suppose an encoder state:

```text id="l2qqcf"
h_7
```

has failed to preserve an important number from the source.

At the decoder timestep that needs this number:

```text id="nzt812"
α_(t,7) = 0.95
```

Will the high attention weight necessarily recover the number?

### Answer

No.

Attention can strongly retrieve:

```text id="eg207w"
h_7
```

but it cannot recreate information that `h_7` does not contain.

Mental model:

```text id="ztmazz"
Encoder
→ stores / represents information

Attention
→ retrieves existing information
```

---

# 🧠 11. Which Problem Is Capacity and Which Is Accessibility?

Classify:

### A.

> The model stored the required information, but the decoder cannot easily reach it.

### B.

> The encoder never represented the required information properly.

### Answer

A:

```text id="plnvl1"
Accessibility problem
```

B:

```text id="9ke9kd"
Representation / capacity problem
```

Attention primarily improves A.

---

# 🧮 12. Compute Global Attention Interaction Count

Suppose:

```text id="rg1e22"
T_src = 200
T_tgt = 50
```

Approximately how many source-target scoring relationships exist?

### Answer

```text id="6uldxr"
T_src × T_tgt
=
200 × 50
=
10,000
```

So there are approximately:

```text id="2tnwb1"
10,000
```

source-target score interactions.

---

# 📐 13. Complexity Reasoning

If classical global attention scores every source position for every target timestep, what is its conceptual attention-scoring complexity?

### Answer

```text id="mmrr5d"
O(T_src × T_tgt)
```

If:

```text id="yaspbc"
T_src ≈ T_tgt ≈ T
```

then this becomes:

```text id="l19afl"
O(T²)
```

But the more precise classical encoder-decoder expression is:

```text id="373ais"
O(T_src T_tgt)
```

---

# 🔍 14. Is Global Attention Always Slow Because It Has Many Comparisons?

A candidate says:

> “Global attention computes many scores, so each score must be processed sequentially.”

Is this correct?

### Answer

No.

At one decoder timestep, many source-position scores can often be computed together using vectorized matrix operations.

For example, Luong-style scoring can compute:

```text id="zf1qfk"
e_t
=
H_enc s_t
```

So:

> many comparisons can still be efficiently parallelized across source positions.

The remaining problem is that decoder timesteps themselves may still be sequential.

---

# 🧠 15. Source-Wise Parallelism vs Target-Wise Sequentiality

Suppose all source states are available.

At decoder timestep `t`, the model scores all source positions at once.

Does this mean:

```text id="sohl01"
t = 1, 2, 3, ..., T_tgt
```

can also all be processed simultaneously?

### Answer

Not in the classical recurrent decoder.

The current decoder state depends on the previous state:

```text id="0z1rid"
s_t
depends on
s_(t-1)
```

So:

```text id="45rkvc"
source comparisons
→ can be vectorized

decoder timesteps
→ remain sequential
```

---

# 📍 16. Local Attention Trade-Off

Suppose global attention considers:

```text id="k1mzyr"
1000 source positions
```

while local attention considers only:

```text id="69a8ou"
21 positions
```

around the predicted alignment.

What is gained and what is lost?

### Answer

### Gain

```text id="qyhig1"
fewer source comparisons
→ lower compute
```

### Risk

```text id="sriz7g"
important source position outside the window
→ inaccessible at that timestep
```

So the trade-off is:

# **Efficiency ↔ Accessibility**

---

# 🧠 17. Why Is Local Attention Not “Free Speedup”?

A developer claims:

> “Use Local Attention. Same information, much less computation.”

What is wrong?

### Answer

Local Attention restricts the candidate source region.

Therefore it does not necessarily provide the same accessibility as global attention.

Relevant information outside the selected region may be missed.

---

# 🧮 18. Dependency Path vs Number of Computations

Model A requires information to pass through:

```text id="sbv1zc"
1 → 2 → 3 → ... → 100
```

Model B directly connects:

```text id="whxbrq"
1 → 100
```

but computes many pairwise relationships.

Which model has the shorter dependency path?

### Answer

Model B.

Direct interaction gives a shorter information path.

But that does **not** necessarily mean Model B uses less computation overall.

This illustrates:

```text id="ea6w7a"
dependency path length
≠
computational complexity
```

---

# 🧠 19. Why Did Self-Attention Become an Attractive Idea?

Complete the progression:

```text id="oscr2e"
Classical Attention
→ decoder directly accesses encoder states

Next Question
→ ?
```

### Answer

```text id="q2a7k6"
Why should source positions
communicate only through recurrence?

Can source positions directly attend
to one another?
```

That leads to:

# **Self-Attention**

---

# 🔍 20. Cross-Attention or Self-Attention?

Classify:

### A.

```text id="hyd672"
decoder state
→ encoder states
```

### B.

```text id="ltbdxx"
source position
→ other source positions
```

### Answer

A:

```text id="dudydu"
Encoder-decoder / cross-attention
```

B:

```text id="l5hwam"
Self-attention
```

---

# ✅ True / False

## 1.

Attention completely removes the recurrent encoder.

**Answer:** False.

---

## 2.

Attention solves the fixed-context bottleneck.

**Answer:** True.

---

## 3.

Attention automatically removes autoregressive decoding.

**Answer:** False.

---

## 4.

Luong scoring can be vectorized across source states.

**Answer:** True.

---

## 5.

Vectorized attention means the full recurrent model is parallel.

**Answer:** False.

---

## 6.

Attention can recreate information absent from the encoder states.

**Answer:** False.

---

## 7.

Global classical attention has approximately `T_src × T_tgt` scoring interactions.

**Answer:** True.

---

## 8.

Local Attention can reduce source accessibility.

**Answer:** True.

---

## 9.

Classical attention eliminates BPTT.

**Answer:** False.

---

## 10.

Shorter dependency path always means lower computational cost.

**Answer:** False.

---

## 11.

Self-attention allows positions within the same sequence to interact directly.

**Answer:** True.

---

## 12.

Transformers remove every possible sequential bottleneck from language generation.

**Answer:** False.

---

# ⭐ Staff Engineer Challenge

You maintain a recurrent document-generation system:

```text id="x670mo"
BiLSTM Encoder
+
Luong Global Attention
+
LSTM Decoder
```

For long documents:

```text id="4v16o7"
T_src ≈ 2000
T_tgt ≈ 300
```

Profiling shows:

* attention score calculation is well vectorized;
* encoder training remains slow;
* decoder inference latency increases linearly with generated length;
* GPU utilization is lower than expected;
* information from distant parts of long documents is sometimes poorly represented.

### Questions

1. Approximately how many source-target attention relationships exist?
2. Why can training still be slow even though attention is vectorized?
3. Why does decoder latency increase with target length?
4. Why might long-range source information still be difficult?
5. What architectural direction would you propose?

---

## Strong Answer

### 1. Attention relationships

```text id="4abhxx"
2000 × 300
=
600,000
```

So approximately:

```text id="bgze50"
600,000
```

source-target scoring relationships exist.

---

### 2. Why is training still slow?

The BiLSTM encoder remains recurrent:

```text id="pbwejm"
h_t depends on h_(t-1)
```

so source sequence positions cannot be fully processed in parallel.

Vectorizing the attention scorer does not remove this dependency.

---

### 3. Why does decoder latency grow with target length?

The LSTM decoder is sequential and autoregressive:

```text id="i84kiu"
generate y_1
↓
generate y_2
↓
...
```

Each token depends on previous decoder computation and generated history.

---

### 4. Why can long-range source information still be hard?

BiLSTM representations still rely on recurrent propagation through many sequence positions.

Attention provides access to each resulting encoder state, but cannot fully eliminate the difficulty of constructing strong long-range recurrent representations.

---

### 5. Architectural direction

Investigate replacing recurrent sequence modeling with:

> self-attention-based sequence interaction

so source positions can interact more directly and training can exploit greater sequence-level parallelism.

This leads naturally to:

# **Transformer Architecture**

---

# 🧠 Final Exercise — Reconstruct the Evolution

Fill in the missing stages:

```text id="geb46o"
Vanilla Seq2Seq
↓
?
↓
Attention
↓
?
↓
Remaining Limitation
↓
?
↓
Transformer
```

### Answer

```text id="wid8tw"
Vanilla Seq2Seq
↓
Fixed Context Bottleneck

Attention
↓
Dynamic Source Access

Remaining Limitation
↓
Recurrence / Sequential Processing

New Idea
↓
Self-Attention

Transformer
```

---

# 🎯 Assignment Mental Model

You should now be able to separate:

```text id="fj6b5l"
What attention fixed
```

from:

```text id="zu4wlw"
What recurrence still caused
```

The full chain is:

```text id="w1iosp"
Fixed Context
↓
Attention
↓
Dynamic Access

but

Recurrent Encoder
+
Recurrent Decoder
↓
Sequential Processing
+
Long Dependency Paths
+
Limited Parallelism

and

Global Attention
↓
Many Pairwise Interactions
```

That leads to the next question:

# **Can attention itself perform sequence modeling without recurrence?**

Answer:

# **Self-Attention**

---

# 🔗 Next Module

This completes the applied work for:

# **06_Attention_Limitations**

The key bridge to remember is:

# **Attention solved access, not recurrence.**

And that is exactly where the next major module begins:

# **04 — Transformer Architecture**

starting with:

# **Why Transformers Were Invented**
