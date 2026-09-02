# 💡 Assignment — Beam Search

> **Goal:** Apply Beam Search by tracing candidate expansion, cumulative scoring, pruning, completed hypotheses, length bias, length normalization, and production trade-offs.

---

# 🧩 Task 1 — Beam Size Meaning

Suppose:

`beam_size = 3`

What does this mean?

### ✅ Answer

It means:

> keep the top 3 partial sequence hypotheses after each pruning step.

It does **not** mean:

> generate exactly 3 output tokens.

---

# 🧠 Task 2 — Beam Size 1

What happens if:

`beam_size = 1`?

### ✅ Answer

Beam Search becomes effectively:

# **Greedy Decoding**

because only the single best hypothesis survives.

---

# 🔄 Task 3 — First Pruning Step

Suppose first-step probabilities are:

```text
A → 0.50
B → 0.30
C → 0.15
D → 0.05
```

Beam size:

`k = 2`

Which hypotheses survive?

### ✅ Answer

Keep:

```text
A → 0.50
B → 0.30
```

Discard:

```text
C
D
```

---

# 🧮 Task 4 — Expand and Score

Current beams:

```text
A → 0.50
B → 0.30
```

Next-step probabilities:

From `A`:

```text
X → 0.40
Y → 0.50
```

From `B`:

```text
X → 0.80
Y → 0.10
```

Calculate the new candidate scores.

### ✅ Answer

From A:

```text
A X = 0.50 × 0.40 = 0.20
A Y = 0.50 × 0.50 = 0.25
```

From B:

```text
B X = 0.30 × 0.80 = 0.24
B Y = 0.30 × 0.10 = 0.03
```

Candidate pool:

```text
A X → 0.20
A Y → 0.25
B X → 0.24
B Y → 0.03
```

---

# ⭐ Task 5 — Prune to Top k

Using the previous candidate pool with:

`beam_size = 2`

which candidates survive?

### ✅ Answer

Top 2:

```text
A Y → 0.25
B X → 0.24
```

This shows the Beam Search loop:

# **Expand → Score → Rank → Prune**

---

# 🚨 Task 6 — Why Greedy Would Miss a Path

Suppose first step:

```text
A → 0.60
B → 0.40
```

Second-step best continuations:

```text
after A → 0.20
after B → 0.90
```

Which complete 2-token path is better?

### ✅ Answer

A-path:

`0.60 × 0.20 = 0.12`

B-path:

`0.40 × 0.90 = 0.36`

So:

# `B-path`

is better.

Greedy would still have committed to A at step 1.

---

# 📐 Task 7 — Sequence Probability

Complete:

# `P(Y|X) = __________________________`

### ✅ Answer

# `Π_t P(y_t | y_<t, X)`

---

# 🧮 Task 8 — Log Score

Suppose a hypothesis has token probabilities:

```text
0.5
0.4
0.8
```

What cumulative log score is used conceptually?

### ✅ Answer

# `log(0.5) + log(0.4) + log(0.8)`

Approximately:

```text
-0.693
-0.916
-0.223
```

Total:

# `-1.832`

Higher log score is better.

So:

`-1.2`

is better than:

`-2.7`.

---

# 🧠 Task 9 — Why Log Space?

Why does Beam Search generally use log probabilities instead of raw products?

### ✅ Answer

Because multiplying many probabilities:

```text
0.7 × 0.8 × 0.6 × ...
```

can become numerically tiny.

Logs convert:

> multiplication into addition.

So:

```text
Π p_t
```

becomes:

```text
Σ log p_t
```

This is more numerically stable.

---

# 🛑 Task 10 — Completed Hypothesis

Suppose a beam becomes:

```text
A B <EOS>
```

What should happen?

### ✅ Answer

It should be marked:

# **completed**

and should not continue expanding.

Wrong:

```text
A B <EOS> C D
```

The completed sequence can still be retained as a final-answer candidate.

---

# 🔄 Task 11 — Active vs Completed

Beam size:

`2`

Current hypotheses:

```text
A B <EOS> → -1.9
A C       → -1.5
```

Classify them.

### ✅ Answer

```text
A B <EOS>
→ completed

A C
→ active
```

Only `A C` should continue normal expansion.

---

# 🚨 Task 12 — Global Optimum Question

A developer says:

> “Beam Search guarantees the highest-probability complete sequence.”

Correct?

### ✅ Answer

No.

Beam Search is:

# **approximate**

because hypotheses are pruned.

Once a path is discarded:

> it cannot return later.

So the true global optimum may still be missed.

---

# 🧠 Task 13 — Search Error vs Model Error

Classify the scenarios.

### Scenario A

The correct sequence has a high model score, but greedy discards it early.

### Scenario B

The correct sequence receives very low probability from the model.

### ✅ Answer

```text
A → Search Error
B → Model Error
```

Beam Search can help more with:

> search error.

It cannot directly fix:

> bad model probabilities.

---

# 📐 Task 14 — Length Bias

Compare:

### Candidate A

```text
A <EOS>
0.8 × 0.8 = 0.64
```

### Candidate B

```text
B C D <EOS>
0.9 × 0.9 × 0.9 × 0.9
```

Calculate Candidate B.

### ✅ Answer

`0.9^4`

# `= 0.6561`

So B wins here.

But notice the general principle:

> every extra token multiplies by another value below 1.

Therefore longer sequences naturally accumulate lower raw probabilities unless their per-token probabilities are very strong.

That creates:

# **length bias**

---

# 🧮 Task 15 — Length Normalization

Suppose:

```text
Candidate A:
log score = -1.0
length = 2

Candidate B:
log score = -1.5
length = 5
```

Using simple average log score:

`score / length`

which candidate is better?

### ✅ Answer

A:

`-1.0 / 2 = -0.5`

B:

`-1.5 / 5 = -0.3`

Since:

`-0.3 > -0.5`

Candidate B becomes better after normalization.

This shows how:

> length normalization can change ranking.

---

# ⚠️ Task 16 — Is Divide-by-Length Always Best?

A developer proposes:

> “Always divide log score by sequence length. Problem solved.”

Correct?

### ✅ Answer

No.

Simple averaging is only one possible strategy.

It may over-reward:

> longer hypotheses.

Real systems may use:

* tuned length penalties
* task-specific normalization

So length handling is:

> a decoding hyperparameter.

---

# 🧠 Task 17 — Each Beam Needs Its Own State

Suppose:

```text
Beam 1 = A B
Beam 2 = A C
```

Can both beams share the same recurrent decoder hidden state?

### ✅ Answer

No.

Different token histories produce:

> different hidden states.

So each beam needs its own:

* hidden state
* cell state for LSTM
* cumulative score
* token history
* completion status

---

# 🚨 Task 18 — Parent Beam Bug

Current beams:

```text
Beam 0 = A
Beam 1 = B
```

After expansion, the best new candidate is:

```text
B X
```

A buggy implementation copies Beam 0's hidden state into `B X`.

What is wrong?

### ✅ Answer

`B X` came from:

# **Beam 1**

So its recurrent state must come from the parent state corresponding to:

`B`.

Using Beam 0's state mixes:

> the wrong decoding history with the new token sequence.

That corrupts future predictions.

---

# 📦 Task 19 — Batch Size vs Beam Size

Suppose:

```text
batch_size = 8
beam_size = 4
```

Roughly how many beam hypotheses may need to be tracked?

### ✅ Answer

`8 × 4`

# `= 32`

hypotheses.

Batch size and beam size are different dimensions.

---

# ⚡ Task 20 — Candidate Count

Suppose:

```text
beam_size = 5
vocabulary_size = 20,000
```

How many candidate token extensions exist conceptually per step before pruning?

### ✅ Answer

`5 × 20,000`

# `= 100,000`

candidate extensions.

Implementations optimize this heavily, but this explains why wider beams are more expensive.

---

# 🚫 Task 21 — Beam Search vs Top-k Sampling

Which description belongs to which?

### A

Keep `k` partial sequences.

### B

Restrict next-token choices to top `k` tokens, then sample.

### ✅ Answer

```text
A → Beam Search
B → Top-k Sampling
```

They are not the same algorithm.

---

# 🧠 Task 22 — Bigger Beam Diagnosis

You test:

```text
beam=2
beam=5
beam=10
```

Quality:

```text
beam 2  → 31.0
beam 5  → 31.8
beam 10 → 31.9
```

Latency:

```text
beam 2  → 100 ms
beam 5  → 170 ms
beam 10 → 300 ms
```

Which setting looks most reasonable from this information alone?

### ✅ Answer

Likely:

# `beam = 5`

because beam 10 gives:

> negligible quality improvement

for:

> substantial additional latency.

This is a production trade-off, not a universal rule.

---

# 🚨 Task 23 — Generic Outputs With Wider Beam

A wider beam starts producing more generic translations.

Does this necessarily mean Beam Search is implemented incorrectly?

### ✅ Answer

No.

The model may assign high probability to:

> generic sequences.

A wider search can become better at finding those high-probability generic paths.

So:

# **better model-score optimization ≠ necessarily better human quality**

---

# 🧠 Task 24 — Does Beam Search Fix Exposure Bias?

### ✅ Answer

No.

Exposure bias concerns:

```text
training
→ ground-truth histories

inference
→ generated histories
```

Beam Search only changes:

> how multiple generated histories are searched.

---

# 🔗 Task 25 — Does Beam Search Fix Context Bottleneck?

### ✅ Answer

No.

Context bottleneck concerns:

> source information access.

Beam Search concerns:

> target sequence search.

Different architectural problems.

---

# ✅ True / False

| #  | Statement                                              | Answer  |
| -- | ------------------------------------------------------ | ------- |
| 1  | Beam Search keeps multiple partial hypotheses.         | ✅ True  |
| 2  | Beam size controls output length.                      | ❌ False |
| 3  | Beam size 1 is effectively greedy search.              | ✅ True  |
| 4  | Beam Search usually uses cumulative sequence scores.   | ✅ True  |
| 5  | Log probabilities are commonly used for scoring.       | ✅ True  |
| 6  | Completed `<EOS>` hypotheses should keep expanding.    | ❌ False |
| 7  | Beam Search guarantees the global optimum.             | ❌ False |
| 8  | Wider beams usually cost more compute and memory.      | ✅ True  |
| 9  | Every beam can safely share the same RNN hidden state. | ❌ False |
| 10 | Raw cumulative probability can favor shorter outputs.  | ✅ True  |
| 11 | Length normalization has no trade-offs.                | ❌ False |
| 12 | Beam Search changes model weights.                     | ❌ False |
| 13 | Beam Search can reduce search error.                   | ✅ True  |
| 14 | Beam Search directly fixes model error.                | ❌ False |
| 15 | Beam Search and top-k sampling are equivalent.         | ❌ False |

---

# ⭐ Staff Engineer Challenge — Production Translation Service

## Scenario

A translation service uses:

```text
LSTM Encoder
↓
Attention
↓
LSTM Decoder
↓
Beam Search
```

Current configuration:

```text
beam_size = 8
length_penalty = default
```

Production observations:

* quality improvement over beam 2 is very small
* P95 latency is much higher
* GPU memory use increased
* some outputs terminate too early
* wider beams produce slightly more generic translations
* 3% of requests hit `max_length`

---

## Question 1 — What would you investigate first?

### ✅ Answer

Compare beam configurations under the same model:

```text
beam 1
beam 2
beam 4
beam 8
```

Measure:

* task quality
* latency
* memory
* generated length
* early `<EOS>` rate
* max-length rate
* repetition
* generic-output frequency

This determines whether wider search actually provides useful value.

---

## Question 2 — Why may beam 8 provide little extra quality over beam 2?

### ✅ Answer

Because beam 2 may already retain most useful high-scoring paths.

After that:

> model error may dominate search error.

So additional search breadth gives diminishing returns.

---

## Question 3 — Why are some outputs too short?

### ✅ Answer

Possible causes:

* cumulative-score length bias
* overly strong `<EOS>` probability
* poorly tuned length penalty
* model training/data distribution

I would not assume beam width alone is the problem.

---

## Question 4 — Should you simply increase the length penalty?

### ✅ Answer

Not blindly.

Increasing length compensation may reduce short outputs, but could also produce:

> unnecessarily long outputs.

Tune it against:

* quality
* output length
* termination behavior

---

## Question 5 — Why might wider beams produce generic outputs?

### ✅ Answer

Because the model may assign high probability to:

> safe/generic hypotheses.

Wider search can find those model-preferred outputs more reliably.

This is a reminder that:

# **Beam Search optimizes model score, not human preference directly.**

---

## Question 6 — What configuration would you likely prefer if beam 4 matches beam 8 quality?

### ✅ Answer

Prefer:

# `beam = 4`

if it provides similar quality with:

* lower latency
* lower memory
* lower compute cost

The smallest beam that meets quality requirements is often the better production choice.

---

## Question 7 — How would you decide whether the remaining problem is search error or model error?

### ✅ Answer

Check whether desirable outputs:

> already receive competitive model scores.

If they exist in wider beams but are pruned under narrow search:

> search error matters.

If desirable outputs remain badly scored even under broad search:

> model error is likely dominant.

---

# 🎯 Final Applied Exercise

Complete the Beam Search pipeline:

```text
Current __________
↓
Expand every __________
↓
Generate candidate __________
↓
Add cumulative __________
↓
__________ candidates
↓
Keep top __________
↓
Separate active and __________ hypotheses
↓
Repeat
```

### ✅ Answer

```text
Current BEAMS
↓
Expand every BEAM
↓
Generate candidate EXTENSIONS
↓
Add cumulative SCORES
↓
RANK candidates
↓
Keep top K
↓
Separate active and COMPLETED hypotheses
↓
Repeat
```

---

# 🧠 Final Assignment Mental Model

Greedy:

```text
A ← survives
B ← gone forever
C ← gone forever
```

Beam Search:

```text
A ─┐
B ─┤ keep k promising paths
C ─┘
```

Then:

```text
A → AX, AY, AZ...
B → BX, BY, BZ...
```

And repeatedly:

# **Expand → Score → Rank → Prune**

But always remember:

```text
Wider Beam
↓
More Exploration
↓
More Compute + Memory + Latency
```

and:

```text
Beam Search
→ helps SEARCH ERROR

Beam Search
≠ fix MODEL ERROR
```

The deepest practical question is not:

> “What is the largest beam we can run?”

It is:

# **“What is the smallest beam that gives enough quality for the latency and cost budget?”**

---

# ⭐ Golden Rule

> **Beam Search improves decoding by delaying commitment and preserving multiple plausible sequence paths, but every extra beam costs resources, and no amount of search can compensate for a model that assigns poor probabilities to the desired output.**
