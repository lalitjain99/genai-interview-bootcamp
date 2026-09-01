# 📝 Revision — Beam Search

> **Goal:** Quickly revise why Beam Search is needed, how it keeps multiple hypotheses alive, how candidates are expanded and pruned, how sequence scores are accumulated, and why length normalization matters.

---

# 🌟 1. Why Beam Search?

Greedy decoding keeps only:

> the highest-probability next token.

Example:

```text
A → 0.55
B → 0.45
```

Greedy chooses:

`A`

and permanently discards:

`B`.

But perhaps:

```text
A-path total = 0.11
B-path total = 0.405
```

So:

# **Best Token Now ≠ Best Complete Sequence**

Beam Search exists to reduce this early-commitment problem.

---

# 🧠 2. Core Idea

Beam Search keeps:

# `k`

high-scoring partial sequences at each decoding step.

`k` is called:

* beam width
* beam size

Example:

```text
beam size = 3
```

means:

> keep 3 active hypotheses after pruning.

---

# ⭐ 3. Greedy vs Beam Search

### Greedy

```text
Keep 1 hypothesis
```

### Beam Search

```text
Keep k hypotheses
```

Important:

# `beam_size = 1`

is effectively:

# **Greedy Decoding**

---

# 🔄 4. Beam Search Cycle

At each timestep:

```text
Current Beams
↓
Expand
↓
Score
↓
Rank
↓
Prune
↓
Keep Top k
↓
Repeat
```

Memorize:

# **Expand → Score → Rank → Prune**

---

# 🧮 5. Simple Example

Beam size:

`k = 2`

First step:

```text
A → 0.55
B → 0.35
C → 0.10
```

Keep:

```text
A → 0.55
B → 0.35
```

Discard:

`C`

---

# 🔄 6. Expand Each Beam

Suppose:

From `A`:

```text
A X → 0.55 × 0.20 = 0.11
A Y → 0.55 × 0.60 = 0.33
```

From `B`:

```text
B X → 0.35 × 0.90 = 0.315
B Y → 0.35 × 0.05 = 0.0175
```

Candidate pool:

```text
A X → 0.11
A Y → 0.33
B X → 0.315
B Y → 0.0175
```

Keep top 2:

```text
A Y → 0.33
B X → 0.315
```

---

# 📐 7. Sequence Score

Raw sequence probability:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

Beam Search usually accumulates:

# **sequence score**

across timesteps.

---

# 📐 8. Why Use Log Probabilities?

Instead of multiplying:

```text
0.8 × 0.7 × 0.9
```

use:

# `log P(Y|X) = Σ_t log P(y_t | y_<t, X)`

Benefits:

* numerically stable
* easier to accumulate
* simpler ranking

---

# 🧮 9. Log Score Intuition

Suppose:

```text
P(A) = 0.5
P(B|A) = 0.4
```

Then:

`P(A B) = 0.2`

Log score:

```text
log(0.5) + log(0.4)
≈ -0.693 + -0.916
≈ -1.609
```

Important:

> higher log score is better.

So:

```text
-1.0 > -3.0
```

---

# ⭐ 10. What Happens to `<EOS>`?

If a hypothesis generates:

```text
A B <EOS>
```

it becomes:

# **completed**

It should not continue expanding.

Beam Search typically tracks:

```text
Active Hypotheses
+
Completed Hypotheses
```

---

# 🛑 11. When Does Beam Search Stop?

Common stopping conditions:

```text
all relevant beams completed
OR
max_length reached
```

Some optimized implementations may stop earlier depending on:

* scores
* finished hypotheses
* length normalization

---

# 🚨 12. Candidate Explosion

If:

```text
beam size = k
vocabulary size = V
```

each decoding step can conceptually generate:

# `k × V`

candidate extensions.

Example:

```text
k = 5
V = 50,000
```

Potential candidates:

# `250,000`

before pruning.

So larger beams cost more.

---

# ⚡ 13. Beam Width Trade-Off

Larger beam:

### Pros

* wider search
* fewer good paths discarded too early

### Cons

* more compute
* more memory
* more latency

So:

```text
beam width ↑
→ search breadth ↑
→ cost ↑
```

---

# 🧠 14. Beam Search Is Approximate

Beam Search prunes hypotheses.

Once removed:

> a path is gone forever.

A discarded sequence might theoretically have become best later.

Therefore:

# **Beam Search does not guarantee the global optimum**

It is an:

> approximate search algorithm.

---

# 🚨 15. Beam Search vs Exhaustive Search

### Exhaustive Search

Explores:

> all possible sequences.

Exact but impossible at realistic scale.

### Beam Search

Explores:

> only top `k` partial sequences.

Much cheaper but approximate.

---

# 📐 16. Why Raw Scores Favor Short Outputs

Each extra token contributes another probability:

`< 1`

or another negative log probability.

Therefore:

```text
longer sequence
→ lower raw cumulative score
```

This can create:

# **length bias**

and make early `<EOS>` attractive.

---

# ⭐ 17. Length Normalization

One simple conceptual correction:

# `normalized_score = log P(Y|X) / length`

This compares something closer to:

> average log probability per token.

But actual implementations may use more sophisticated:

> length penalties.

---

# ⚠️ 18. Length Penalty Is a Trade-Off

Too little compensation:

> outputs may be too short.

Too much compensation:

> outputs may become too long.

So length penalty is a:

# **decoding hyperparameter**

---

# 🧠 19. Search Error vs Model Error

This distinction is extremely important.

### Search Error

The model gives a good sequence high probability, but decoding fails to find it.

Beam Search can help.

### Model Error

The model assigns low probability to the correct sequence.

Beam Search cannot fix that.

So:

```text
Beam Search
→ reduces search error

not necessarily
→ model error
```

---

# ⭐ 20. Beam Search Does Not Change Model Probabilities

The model still produces:

# `P(y_t | y_<t, X)`

Beam Search only changes:

> how candidate sequences are explored.

So:

```text
Model
→ probabilities

Beam Search
→ search
```

---

# 🧠 21. Every Beam Needs Its Own State

Different hypotheses have different histories.

Example:

```text
Beam 1:
A B

Beam 2:
A C
```

Therefore their decoder states differ:

```text
d^(1) ≠ d^(2)
```

Beam Search must maintain:

* separate token history
* separate decoder state
* separate cumulative score

for each beam.

---

# 🚨 22. Parent Beam Tracking

Suppose:

```text
Beam 0 → A
Beam 1 → B
```

and best next candidate becomes:

```text
B X
```

The decoder state must come from:

> Beam 1

not Beam 0.

So implementations need:

> parent-beam bookkeeping.

---

# 🧠 23. Beam Size ≠ Sequence Length

If:

```text
beam size = 5
```

it means:

> keep 5 candidate sequences.

It does not mean:

> output contains 5 tokens.

These are unrelated concepts.

---

# 🚫 24. Beam Search ≠ Top-k Sampling

### Beam Search

`k` means:

> number of hypotheses retained.

### Top-k Sampling

`k` means:

> number of token options allowed before random sampling.

So:

```text
Beam k
→ sequence hypotheses

Top-k
→ token choices
```

---

# 🧠 25. Beam Search vs Sampling

### Beam Search

Goal:

> find high-scoring sequences.

Usually deterministic for fixed conditions.

### Sampling

Goal:

> generate stochastically from the model distribution.

Useful when diversity matters.

---

# ⭐ 26. Why Translation Often Uses Beam Search

Translation usually prioritizes:

* fidelity
* consistency
* one strong output

So Beam Search is a natural fit.

Open-ended generation often prioritizes:

* creativity
* diversity

where sampling may be more appropriate.

---

# ⚠️ 27. Bigger Beam Is Not Always Better

A larger beam searches the model more thoroughly.

But:

> better search of a flawed probability distribution does not guarantee better human-quality output.

A wider beam can even prefer:

* generic outputs
* short outputs
* overconfident but undesirable sequences

So:

# **better search ≠ automatically better task quality**

---

# 📦 28. Batch Size vs Beam Size

### Batch Size

Number of source examples.

### Beam Size

Candidate hypotheses:

> per source example.

Example:

```text
batch size = 8
beam size = 5
```

may require tracking roughly:

```text
8 × 5 = 40
```

hypotheses.

---

# 🔗 29. Beam Search Does Not Solve Other Problems

Beam Search does **not** directly solve:

### Context Bottleneck

Source-access problem.

### Exposure Bias

Training/inference-history mismatch.

### Model Error

Bad probability distribution.

Beam Search solves:

# **target-sequence search**

---

# 🔗 30. Problem Taxonomy

```text
Context Bottleneck
→ source access

Teacher Forcing
→ training history

Exposure Bias
→ train/inference mismatch

Autoregressive Decoding
→ feedback generation

Greedy Search
→ keep 1 hypothesis

Beam Search
→ keep k hypotheses

Attention
→ dynamic source access
```

---

# 🚫 31. High-Yield Traps

### ❌ Beam Search changes model probabilities

No.

### ❌ Beam width = output length

No.

### ❌ Beam size 1 is fundamentally different from greedy

No.

### ❌ Beam Search keeps top `k` tokens only

No.

Top `k` partial sequences.

### ❌ Beam Search guarantees global optimum

No.

### ❌ Larger beam is always better

No.

### ❌ `<EOS>` hypotheses should keep expanding

No.

### ❌ Raw cumulative score treats all lengths fairly

No.

### ❌ Length normalization always fixes scoring

No.

### ❌ Beam Search fixes model error

No.

### ❌ Beam Search fixes exposure bias

No.

### ❌ Beam Search = top-k sampling

No.

---

# 📐 32. Formula Flash Card

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Log Score

# `S(Y) = Σ_t log P(y_t | y_<t, X)`

### Beam Rule

# `keep top k partial hypotheses`

### Generic Length Adjustment

# `S_norm(Y) = S(Y) / LP(Y)`

### Beam Size 1

# `Beam(k=1) ≈ Greedy`

---

# 🎤 33. 30-Second Interview Answer

> **Beam Search is an approximate decoding algorithm that keeps the top `k` partial sequence hypotheses at every autoregressive decoding step instead of committing immediately to one path like greedy search. Each active beam is expanded with possible next tokens, cumulative scores—typically log probabilities—are updated, all candidates are ranked, and only the best `k` survive. Beam Search reduces search errors caused by early greedy decisions, but it increases compute and memory and still does not guarantee the globally optimal sequence because pruned paths cannot recover. Practical systems also often use length normalization because raw cumulative scores naturally favor shorter outputs.**

---

# 🧠 10 Things You Must Know Cold

1. **Greedy keeps 1 hypothesis.**
2. **Beam Search keeps `k` hypotheses.**
3. **Beam size 1 ≈ greedy.**
4. **Core loop = Expand → Score → Rank → Prune.**
5. **Cumulative log probability is commonly used.**
6. **`<EOS>` creates a completed hypothesis.**
7. **Beam Search is approximate, not exact.**
8. **Raw scores can favor shorter sequences.**
9. **Length penalty handles length bias.**
10. **Beam Search helps search error, not model error.**

---

# 🧠 Final Mental Model

Greedy:

```text
A ← keep
B ← discard
C ← discard
```

Beam Search:

```text
A ─┐
B ─┤ keep top k
C ─┘
```

Then:

```text
A → AX, AY, AZ...
B → BX, BY, BZ...
```

Pool all candidates:

```text
AX
AY
AZ
BX
BY
BZ
...
```

Then:

# **Rank → Keep Top k → Repeat**

Full loop:

```text
Current Beams
↓
Expand
↓
Cumulative Scores
↓
Rank
↓
Prune
↓
Track Active + Completed
↓
Repeat
```

The central trade-off:

```text
Wider Beam
→ better exploration
→ higher compute and latency
```

And the key limitation:

# **Beam Search improves target search, but it still does not solve the fixed source-access problem.**

That is why our next chapter is:

# `14_Attention_Motivation`

---

# ⭐ Golden Rule

> **Beam Search delays commitment: instead of betting everything on one locally best token, it keeps several promising sequence paths alive—but because it still prunes, it trades computational cost for better approximate sequence-level search rather than guaranteeing the global optimum.**
