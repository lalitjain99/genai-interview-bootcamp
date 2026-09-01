# ⚡ Cheat Sheet — Beam Search

> **Goal:** Fast revision of why Beam Search is needed, how it keeps multiple sequence hypotheses alive, how cumulative log scores are used, why length bias matters, and what trade-offs matter in production.

---

# 🌟 1. Beam Search in One Line

> **Keep the top `k` partial sequences at each decoding step instead of committing to only one path.**

```text
Greedy
→ keep 1 path

Beam Search
→ keep k paths
```

`k` = beam size / beam width.

---

# 🧠 2. Why Beam Search Exists

Greedy decoding optimizes:

# `best token now`

But sequence generation wants:

# `best complete sequence`

So:

# **Best Token Now ≠ Best Sequence Overall**

Beam Search delays commitment.

---

# ⭐ 3. Beam Size

If:

`beam_size = 4`

then:

> keep 4 partial sequence hypotheses after pruning.

Important:

# `beam_size = 1 ≈ greedy decoding`

---

# 🔄 4. Core Beam Search Loop

Memorize:

# **Expand → Score → Rank → Prune**

Full flow:

```text
Current Beams
↓
Expand Every Beam
↓
Generate Candidate Extensions
↓
Update Scores
↓
Rank
↓
Keep Top k
↓
Repeat
```

---

# 🧮 5. Simple Example

Beam size:

`k = 2`

First step:

```text
A → 0.50
B → 0.35
C → 0.15
```

Keep:

```text
A
B
```

Then expand both:

```text
A → AX, AY...
B → BX, BY...
```

Pool everything.

Rank everything.

Keep top 2.

---

# 📐 6. Sequence Probability

For sequence:

`Y = (y_1, ..., y_T)`

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

Beam Search ranks hypotheses using:

> cumulative sequence score.

---

# 📐 7. Use Log Probabilities

Instead of:

`0.8 × 0.7 × 0.9`

use:

# `Σ_t log P(y_t | y_<t, X)`

So:

# `S(Y) = Σ_t log P(y_t | y_<t, X)`

Benefits:

* avoids numerical underflow
* converts multiplication to addition
* easy score accumulation

---

# 🧠 8. Log Score Rule

Log probabilities are usually negative.

Example:

```text
Candidate A → -1.2
Candidate B → -2.5
```

Better candidate:

# `A`

because:

# `-1.2 > -2.5`

So remember:

> **less negative = better**

---

# 🛑 9. `<EOS>` Handling

If hypothesis becomes:

```text
A B <EOS>
```

it is:

# **completed**

Do not expand it further.

Beam Search commonly tracks:

```text
Active Hypotheses
+
Completed Hypotheses
```

---

# 🚨 10. Why Beam Search Is Approximate

Beam Search prunes candidates.

Once discarded:

> a path cannot return.

Therefore:

# **Beam Search does not guarantee the global optimum**

It is:

> approximate search.

---

# 🧠 11. Beam Search vs Exhaustive Search

### Exhaustive Search

Explore:

> every possible sequence.

Exact but infeasible.

### Beam Search

Explore:

> only top `k` partial sequences.

Practical but approximate.

---

# ⚡ 12. Beam Width Trade-Off

Larger beam:

```text
Search breadth ↑
Compute ↑
Memory ↑
Latency ↑
```

Potential benefit:

> fewer good paths pruned early.

But:

# **larger beam is not always better**

---

# 🚨 13. Length Bias

Each extra token multiplies another probability `< 1`.

So:

```text
longer sequence
→ lower raw probability
```

or in log space:

```text
longer sequence
→ more negative cumulative score
```

This can favor:

> short outputs / early `<EOS>`.

---

# 📐 14. Length Normalization

Simple conceptual version:

# `normalized_score = log P(Y|X) / length`

Generic form:

# `S_norm(Y) = S(Y) / LP(Y)`

where:

`LP(Y)`

is a length-penalty function.

Purpose:

> reduce unfair preference for short sequences.

---

# ⚠️ 15. Length Penalty Is a Hyperparameter

Too weak:

> outputs may be too short.

Too strong:

> outputs may become too long.

So length handling must be:

> tuned empirically.

---

# 🧠 16. Search Error vs Model Error

### Search Error

Good sequence has high model probability, but decoder fails to find it.

Beam Search can help.

### Model Error

Desired sequence has low model probability.

Beam Search cannot fix that.

Remember:

# **Beam Search improves search, not the model itself.**

---

# ⭐ 17. Beam Search Does Not Change Model Probabilities

Model:

# `P(y_t | y_<t, X)`

Beam Search:

> decides how to explore those probabilities.

So:

```text
Model
→ scoring

Beam Search
→ search
```

---

# 🧠 18. Every Beam Needs Its Own State

Different histories:

```text
Beam 1 = A B
Beam 2 = A C
```

produce different decoder states.

Each beam needs its own:

* token history
* cumulative score
* hidden state
* LSTM cell state if used
* completion status
* parent beam information

---

# 🚨 19. Parent Beam Tracking

If:

```text
Beam 0 = A
Beam 1 = B
```

and best new candidate is:

```text
B X
```

its decoder state must come from:

> Beam 1.

Wrong parent state = corrupted decoding history.

---

# 📦 20. Batch Size vs Beam Size

### Batch Size

Number of source examples.

### Beam Size

Number of hypotheses per source example.

Example:

```text
batch = 8
beam = 5
```

roughly:

```text
8 × 5 = 40
```

hypotheses to track.

---

# 🚫 21. Beam Search ≠ Top-k Sampling

### Beam Search

`k` = number of sequence hypotheses kept.

### Top-k Sampling

`k` = number of token choices allowed before sampling.

So:

```text
Beam k
→ hypotheses

Top-k sampling
→ token candidates
```

---

# 🎲 22. Beam Search vs Sampling

### Beam Search

Goal:

> high-scoring sequences.

Usually more deterministic.

### Sampling

Goal:

> stochastic/diverse generation.

Often better when:

* diversity matters
* many outputs are valid

---

# 🧠 23. Why Bigger Beam Can Hurt

A wider beam searches model probabilities more thoroughly.

But if the model prefers:

* generic outputs
* short outputs
* undesirable high-probability sequences

then wider search may find them more reliably.

So:

# **Better Search ≠ Better Human Quality**

---

# 🔗 24. Beam Search Does Not Solve Other Problems

Beam Search does not directly solve:

```text
Context Bottleneck
→ source access

Exposure Bias
→ training/inference mismatch

Model Error
→ poor probability assignment
```

Beam Search solves:

# **target-sequence search**

---

# 📐 25. Complexity Intuition

With:

* beam size = `k`
* vocabulary = `V`

each step conceptually considers up to:

# `k × V`

extensions.

Across output length `T`:

roughly:

# `O(T × k × V)`

for candidate scoring/search logic, ignoring model-specific optimizations.

---

# 🚫 26. High-Yield Traps

### ❌ Beam Search keeps top `k` tokens

No.

Top `k` partial sequences.

### ❌ Beam size = output length

No.

### ❌ Beam size 1 is different from greedy

Effectively no.

### ❌ Beam Search guarantees global optimum

No.

### ❌ Bigger beam is always better

No.

### ❌ `<EOS>` hypotheses should keep expanding

No.

### ❌ Raw cumulative score is length-neutral

No.

### ❌ Length normalization always solves length bias perfectly

No.

### ❌ Beam Search changes model probabilities

No.

### ❌ Beam Search fixes model error

No.

### ❌ Beam Search fixes exposure bias

No.

### ❌ Beam Search = top-k sampling

No.

---

# 📐 27. Formula Flash Card

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Cumulative Log Score

# `S(Y) = Σ_t log P(y_t | y_<t, X)`

### Beam Rule

# `keep top k partial hypotheses`

### Length-Adjusted Score

# `S_norm(Y) = S(Y) / LP(Y)`

### Beam Size 1

# `Beam(k=1) ≈ Greedy`

---

# 🎤 28. 30-Second Interview Answer

> **Beam Search is an approximate decoding algorithm that keeps the top `k` partial sequence hypotheses at each autoregressive decoding step instead of committing to one path like greedy decoding. Each beam is expanded, cumulative scores—usually log probabilities—are updated, all candidates are ranked, and only the best `k` survive. This reduces search errors caused by early greedy decisions, but it increases compute, memory, and latency. Because hypotheses are pruned, Beam Search is not guaranteed to find the global optimum, and practical systems often use length normalization because raw cumulative scores can favor short sequences.**

---

# 🧠 10 Things You Must Know Cold

1. **Greedy keeps 1 path; Beam Search keeps `k`.**
2. **Beam size 1 ≈ greedy.**
3. **Core loop = Expand → Score → Rank → Prune.**
4. **Cumulative log probabilities are commonly used.**
5. **`<EOS>` marks a completed hypothesis.**
6. **Beam Search is approximate.**
7. **Wider beams cost more compute and memory.**
8. **Raw scores may favor shorter sequences.**
9. **Length penalty adjusts length bias.**
10. **Beam Search helps search error, not model error.**

---

# 🧠 Final Mental Model

Greedy:

```text
Step 1

A ← keep
B ← discard
C ← discard
```

Beam Search:

```text
Step 1

A ─┐
B ─┤ keep top k
C ─┘
```

Then:

```text
A → AX, AY, AZ...
B → BX, BY, BZ...
```

Pool:

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
Score
↓
Rank
↓
Prune
↓
Track Active + Completed
↓
Repeat
```

Main trade-off:

```text
Beam Width ↑
→ More Exploration
→ More Compute
→ More Memory
→ More Latency
```

And the deepest distinction:

```text
Beam Search
→ better SEARCH

Model
→ determines WHAT gets high probability
```

So:

# **better search cannot repair bad model probabilities.**

---

# ⭐ Golden Rule

> **Beam Search delays commitment by keeping several promising sequence paths alive, but every extra beam costs compute and every pruned path is permanently lost—so Beam Search is a quality-versus-latency search strategy, not a guarantee of the best sequence.**
