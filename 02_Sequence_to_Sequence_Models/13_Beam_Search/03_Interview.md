# 🎤 Interview Questions — Beam Search

> **Goal:** Test whether you understand why Beam Search is needed, how it differs from greedy decoding, how cumulative sequence scores are computed, why pruning makes it approximate, how `<EOS>` and length bias are handled, and what practical trade-offs matter in production.

---

# 🟢 Beginner

## Q1. What is Beam Search?

### ✅ Answer

Beam Search is an approximate decoding algorithm that keeps:

> the top `k` partial sequence hypotheses

at each decoding step instead of keeping only one path.

Here:

`k`

is called:

# **beam size / beam width**

Conceptually:

```text id="beam-int-1"
Current Beams
↓
Expand
↓
Score
↓
Rank
↓
Keep Top k
↓
Repeat
```

---

## Q2. Why do we need Beam Search if greedy decoding already works?

### ✅ Answer

Greedy decoding chooses:

> the highest-probability token at the current step.

But the locally best token may not lead to:

> the highest-scoring complete sequence.

Example:

```text id="beam-int-2"
A → 0.55
B → 0.45
```

Greedy chooses A.

But if:

```text id="beam-int-3"
best continuation after A = 0.20
best continuation after B = 0.90
```

then:

```text id="beam-int-4"
A-path = 0.55 × 0.20 = 0.11
B-path = 0.45 × 0.90 = 0.405
```

So Beam Search keeps alternatives alive longer.

---

## Q3. What does beam size mean?

### ✅ Answer

Beam size tells us:

> how many partial sequence hypotheses survive after pruning at each decoding step.

Example:

```text id="beam-int-5"
beam size = 4
```

means:

> keep 4 candidate sequences.

It does **not** mean:

> generate 4 tokens.

---

## Q4. What happens when beam size is 1?

### ✅ Answer

# **Beam Search with beam size 1 is effectively greedy decoding.**

Only the single highest-scoring hypothesis survives after each step.

---

# 🟡 Intermediate

## Q5. What is the core Beam Search loop?

### ✅ Answer

The main loop is:

# **Expand → Score → Rank → Prune**

More explicitly:

```text id="beam-int-6"
1. Take current beams
2. Expand each with possible next tokens
3. Update cumulative sequence scores
4. Pool all candidates
5. Rank them
6. Keep top k
7. Repeat
```

---

## Q6. How are Beam Search candidates scored?

### ✅ Answer

A partial sequence can be scored using cumulative sequence probability:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

In practice, we usually use cumulative log probability:

# `S(Y) = Σ_t log P(y_t | y_<t, X)`

because log scores are numerically more stable.

---

## Q7. Why are log probabilities preferred?

### ✅ Answer

Raw sequence probability multiplies many numbers below 1:

```text id="beam-int-7"
0.8 × 0.7 × 0.9 × ...
```

For long sequences, this becomes extremely small.

Taking logs converts:

> multiplication into addition.

So:

# `log P(Y|X) = Σ_t log P(y_t | y_<t, X)`

This avoids underflow and simplifies score accumulation.

---

## Q8. What happens when a beam generates `<EOS>`?

### ✅ Answer

That hypothesis becomes:

# **completed**

It should no longer be expanded.

Beam Search often tracks:

```text id="beam-int-8"
Active Hypotheses
+
Completed Hypotheses
```

Other active hypotheses may continue decoding.

---

## Q9. Is Beam Search guaranteed to find the globally best sequence?

### ✅ Answer

No.

Beam Search prunes hypotheses at every step.

Once a path is removed:

> it cannot recover later.

So a path that looks weak early but would have become optimal later may be lost.

Therefore Beam Search is:

# **approximate search**

not exhaustive search.

---

## Q10. What is the difference between Beam Search and exhaustive search?

### ✅ Answer

Exhaustive search evaluates:

> all possible output sequences.

Beam Search keeps only:

> the best `k` partial sequences.

So:

```text id="beam-int-9"
Exhaustive Search
→ exact but infeasible

Beam Search
→ approximate but practical
```

---

# 🔴 Advanced

## Q11. Why can Beam Search prefer short sequences?

### ✅ Answer

Every additional token contributes another probability less than 1.

In log space, each token usually adds:

> another negative log probability.

Therefore:

```text id="beam-int-10"
longer sequence
→ more negative cumulative score
```

This can make short hypotheses and early `<EOS>` overly attractive.

This is:

# **length bias**

---

## Q12. What is length normalization or a length penalty?

### ✅ Answer

Length normalization adjusts cumulative sequence scores to reduce unfair preference for short sequences.

A simple conceptual version is:

# `normalized_score = log P(Y|X) / length`

A more generic form is:

# `S_norm(Y) = S(Y) / LP(Y)`

where:

`LP(Y)`

depends on sequence length.

Exact formulas vary by implementation.

---

## Q13. Can a larger beam ever make output quality worse?

### ✅ Answer

Yes.

A larger beam searches the model's probability distribution more thoroughly.

But if the model itself prefers:

* short outputs
* generic outputs
* repetitive outputs
* undesirable high-probability sequences

then a wider beam may become better at finding those undesirable sequences.

So:

# **better search of the model does not necessarily mean better human-quality output.**

---

## Q14. What is the difference between search error and model error?

### ✅ Answer

### Search Error

The model assigns a good sequence high probability, but decoding fails to find it.

Beam Search can help.

### Model Error

The model itself assigns low probability to the desired sequence.

Beam Search cannot fix that.

So:

```text id="beam-int-11"
Beam Search
→ helps search error

not necessarily
→ model error
```

---

## Q15. Why does each beam need its own decoder state?

### ✅ Answer

Different beams represent different generated histories.

Example:

```text id="beam-int-12"
Beam 1 = A B
Beam 2 = A C
```

Because the previous tokens differ:

> their recurrent hidden states differ.

So each hypothesis needs its own:

* token history
* cumulative score
* decoder hidden state
* LSTM cell state if applicable
* completion status

Sharing one evolving state across all beams would be incorrect.

---

# ⭐ Staff Engineer Challenge

## Scenario

You deploy a recurrent Seq2Seq translation service.

Current configuration:

```text id="beam-int-13"
Decoder:
LSTM

Search:
Beam Search

Beam Width:
8
```

Production observations:

* quality is only slightly better than beam width 2
* P95 latency increased significantly
* memory usage increased
* outputs are sometimes too short
* wider beams occasionally produce more generic translations
* around 4% of outputs terminate unusually early

---

## Question 1. Why did latency increase?

### ✅ Answer

Beam Search maintains multiple active hypotheses.

With beam width 8, the system evaluates:

> multiple alternative decoding histories per input.

Conceptually:

```text id="beam-int-14"
beam width ↑
↓
active states ↑
↓
candidate expansions ↑
↓
compute + memory ↑
↓
latency ↑
```

---

## Question 2. Why may quality gains saturate as beam size increases?

### ✅ Answer

Once the beam is wide enough to retain the useful high-scoring paths:

> further widening may mostly preserve weaker alternatives.

At that point:

* search error is already low
* model error may dominate
* additional search gives diminishing returns

So wider beam does not guarantee proportional quality improvement.

---

## Question 3. What could explain unusually short outputs?

### ✅ Answer

One likely factor is:

# **length bias**

Since cumulative log probability becomes more negative as sequence length grows, Beam Search may favor:

> early `<EOS>`

unless scoring is adjusted appropriately.

I would inspect:

* `<EOS>` scores
* generated length distribution
* length normalization / penalty
* training data length distribution
* whether the model itself overpredicts `<EOS>`

---

## Question 4. Would increasing beam width fix early termination?

### ✅ Answer

Not necessarily.

If the model strongly prefers:

`<EOS>`

or the scoring function favors short sequences, a wider beam may actually preserve or amplify that behavior.

The issue should be diagnosed at:

> model scoring + length scoring

rather than assuming beam width is the solution.

---

## Question 5. How would you tune Beam Search in production?

### ✅ Answer

I would compare configurations such as:

```text id="beam-int-15"
beam = 1
beam = 2
beam = 4
beam = 8
```

while measuring:

* task quality
* latency
* memory
* average output length
* early `<EOS>` rate
* max-length rate
* repetition
* cost per request

Then choose the:

> smallest beam that provides meaningful quality improvement within system constraints.

---

## Question 6. How would you distinguish search error from model error?

### ✅ Answer

I would inspect whether desirable outputs already receive:

> strong model scores

but are lost because decoding prunes them.

If increasing beam width significantly recovers those outputs:

> search error was likely important.

If the desired outputs remain low-scoring even with wider search:

> model error is more likely.

---

## Question 7. Why can wider Beam Search produce more generic outputs?

### ✅ Answer

If the model assigns high probability to generic sequences, wider search can become more effective at locating those:

> high-probability generic paths.

Beam Search optimizes model score.

It does not optimize:

> human preference automatically.

So a production system may need:

* tuned beam width
* length penalty
* task metrics
* reranking
* constraints

depending on the application.

---

# ⚡ Rapid-Fire Traps

### ❌ “Beam Search keeps the top `k` tokens.”

False.

It keeps the top `k`:

> partial sequences.

---

### ❌ “Beam size 5 means the output has 5 tokens.”

False.

---

### ❌ “Beam Search changes the model's probabilities.”

False.

---

### ❌ “Beam Search always finds the global optimum.”

False.

---

### ❌ “Beam size 1 is unrelated to greedy decoding.”

False.

It is effectively greedy.

---

### ❌ “Larger beams are always better.”

False.

---

### ❌ “Completed `<EOS>` beams should continue expanding.”

False.

---

### ❌ “Raw cumulative log score is length-neutral.”

False.

---

### ❌ “Length normalization is always guaranteed to improve outputs.”

False.

---

### ❌ “Beam Search fixes exposure bias.”

False.

---

### ❌ “Beam Search fixes the context-vector bottleneck.”

False.

---

### ❌ “Beam Search and top-k sampling are the same.”

False.

---

# 📐 Formula Flash Card

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Log Sequence Score

# `S(Y) = Σ_t log P(y_t | y_<t, X)`

### Beam Rule

# `Keep top k partial hypotheses at each step`

### Generic Length Adjustment

# `S_norm(Y) = S(Y) / LP(Y)`

### Beam Size 1

# `Beam(k=1) ≈ Greedy`

---

# 🎤 30-Second Interview Answer

> **Beam Search is an approximate decoding algorithm for autoregressive sequence models that keeps the top `k` partial sequence hypotheses at each timestep rather than committing to one path like greedy decoding. Each hypothesis is expanded, cumulative scores—usually log probabilities—are updated, candidates are ranked, and only the best `k` survive. This reduces search errors from early greedy decisions, but Beam Search increases compute, memory, and latency and still cannot guarantee the globally best sequence because pruned paths are permanently lost. Practical systems also often apply length normalization because cumulative scores naturally favor shorter outputs.**

---

# 🧠 Final Interview Mental Model

Think of Beam Search as:

```text id="beam-int-16"
Greedy:
Keep 1 path

Beam:
Keep k paths
```

Then repeatedly:

```text id="beam-int-17"
Current Beams
↓
Expand Every Beam
↓
Generate Candidate Extensions
↓
Add Cumulative Scores
↓
Rank
↓
Keep Top k
↓
Mark <EOS> Hypotheses Completed
↓
Repeat
```

And always distinguish:

```text id="beam-int-18"
Model
→ defines probabilities

Beam Search
→ searches those probabilities
```

Beam Search helps when:

```text id="beam-int-19"
good path exists
but
greedy search misses it
```

It does not fix:

```text id="beam-int-20"
model assigns bad scores
```

The deepest interview insight is:

# **better search and better modeling are not the same problem.**

---

# ⭐ Golden Rule

> **Beam Search buys more exploration by delaying commitment, but it never escapes the model's own probability distribution: it can reduce search error, not magically repair model error, and every increase in beam width must justify its additional latency and memory cost.**
