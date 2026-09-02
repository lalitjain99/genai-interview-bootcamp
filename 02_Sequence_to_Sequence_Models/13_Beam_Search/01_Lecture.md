# 📘 Lecture — Beam Search

## 🌟 Introduction

In the previous lecture, we studied **Inference and Autoregressive Decoding**.

We saw that a decoder generates:

```text
<BOS>
↓
token 1
↓
token 2
↓
token 3
↓
...
↓
<EOS>
```

At each step, the model produces a probability distribution.

Then a decoding strategy decides:

> which token should be selected?

The simplest strategy was:

# **Greedy Decoding**

Greedy decoding chooses the highest-probability token at every step.

For example:

```text
A → 0.55
B → 0.45
```

Greedy chooses:

`A`

and immediately discards:

`B`.

But we discovered a problem.

Suppose the continuation probabilities are:

```text
After A:
best continuation = 0.20

After B:
best continuation = 0.90
```

Then:

```text
A-path = 0.55 × 0.20 = 0.11

B-path = 0.45 × 0.90 = 0.405
```

So the sequence starting with the lower-probability first token is actually much better overall.

This exposes the central limitation of greedy decoding:

# **Best Token Now ≠ Best Sequence Overall**

So the obvious next engineering question is:

> What if we do not commit to only one candidate immediately?

Instead:

> What if we keep several promising partial sequences alive?

That idea gives us:

# **Beam Search**

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to:

## 🧠 Conceptual Understanding

* Explain why greedy decoding can fail.
* Explain what Beam Search is.
* Understand beam width.
* Understand candidate expansion and pruning.
* Explain why Beam Search is an approximation to sequence search.
* Understand the quality vs compute trade-off.

## 📐 Mathematical Understanding

* Score partial sequences using log probabilities.
* Understand cumulative sequence score.
* Understand why raw probability decreases with sequence length.
* Understand length bias.
* Understand length normalization conceptually.

## 🧮 Practical Understanding

* Trace Beam Search step by step.
* Compare beam size 1, 2, and larger beams.
* Handle `<EOS>` candidates.
* Understand completed vs active beams.
* Diagnose poor Beam Search behavior.

## 🔗 Architectural Understanding

* Connect:

  * autoregressive decoding
  * greedy search
  * Beam Search
  * sequence probability
  * length normalization
* Prepare for the next lecture:

  * **Attention Motivation**

---

# 📖 Part 1 — Revisit Greedy Search

Suppose the first decoding step gives:

```text
A → 0.55
B → 0.45
```

Greedy keeps:

```text
A
```

and permanently discards:

```text
B
```

Now suppose the second-step distributions are:

After `A`:

```text
X → 0.20
Y → 0.10
```

After `B`:

```text
X → 0.90
Y → 0.05
```

The best sequence through `A`:

`0.55 × 0.20 = 0.11`

The best sequence through `B`:

`0.45 × 0.90 = 0.405`

Greedy lost the better sequence because:

> it committed too early.

---

# 🧠 Part 2 — What Do We Actually Want?

During decoding, we do not truly want:

# `highest-probability next token`

We want:

# `highest-scoring complete output sequence`

Conceptually:

`Y* = argmax_Y P(Y|X)`

But the number of possible sequences is enormous.

If vocabulary size is:

`V`

and output length is:

`T`

then there are roughly:

# `V^T`

possible token sequences.

For:

```text
V = 10,000
T = 20
```

the search space is astronomical.

We cannot evaluate every possible sequence.

---

# ⭐ Part 3 — Two Extremes

We now have two possible strategies.

## Extreme 1 — Greedy

Keep only:

```text
1 candidate
```

Cheap.

But may discard good paths too early.

## Extreme 2 — Exhaustive Search

Keep:

```text
every possible sequence
```

Best search quality in principle.

But computationally impossible.

We need something between them.

---

# 🧠 Part 4 — Beam Search Intuition

Beam Search says:

> At every decoding step, keep only the `k` most promising partial sequences.

Where:

`k`

is called:

# **beam width / beam size**

For example:

`beam_size = 2`

means:

> keep two candidate sequences alive after every pruning step.

So:

```text
Greedy
→ keep 1 path

Beam Search k=2
→ keep 2 paths

Beam Search k=5
→ keep 5 paths
```

---

# ⭐ Part 5 — Beam Size 1

Important:

# **Beam Search with beam size 1 is effectively Greedy Search**

Because after every expansion:

> only the single best candidate survives.

This is a common interview question.

---

# 📖 Part 6 — First Beam Search Example

Suppose:

`beam_size = 2`

At first step:

```text
A → 0.55
B → 0.35
C → 0.10
```

Keep top 2:

```text
A → 0.55
B → 0.35
```

Discard:

```text
C
```

Now we have:

# two active beams

---

# 🔄 Part 7 — Expand Every Active Beam

At next step:

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

Now candidate pool:

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

This is the key Beam Search cycle:

# **Expand → Score → Rank → Prune**

---

# 🧠 Part 8 — Why Beam Search Is Better Than Greedy

Greedy would have kept only:

```text
A
```

after step 1.

Then it could never discover:

```text
B X
```

Beam Search temporarily keeps both:

```text
A
B
```

so the second token can change which path becomes best.

That is the entire advantage.

---

# 📐 Part 9 — Sequence Score

For candidate sequence:

`Y = (y_1, ..., y_t)`

the raw sequence probability is:

# `P(Y|X) = Π_i P(y_i | y_<i, X)`

For Beam Search, we score each partial hypothesis using:

> cumulative probability

or more commonly:

> cumulative log probability.

---

# 📐 Part 10 — Why Log Scores?

Instead of:

`0.55 × 0.60 × 0.80 ...`

use:

# `log P(Y|X) = Σ_i log P(y_i | y_<i, X)`

Why?

* avoids numerical underflow
* converts multiplication to addition
* makes score accumulation easier

So practical Beam Search usually stores:

# **cumulative log-probability score**

---

# 🧮 Part 11 — Log Score Example

Suppose:

```text
P(A) = 0.5
P(B|A) = 0.4
```

Then:

`P(A B) = 0.5 × 0.4 = 0.2`

Log score:

`log 0.5 + log 0.4`

≈

`-0.693 + -0.916`

# `= -1.609`

Higher log score is better.

Because:

```text
-1.0 > -3.0
```

So remember:

> log probabilities are usually negative, and less negative is better.

---

# ⭐ Part 12 — Beam Search Algorithm

At each target step:

```text
1. Take current beams
2. Run decoder for each beam
3. Generate candidate next-token scores
4. Add token scores to cumulative beam scores
5. Pool all expansions
6. Rank candidates
7. Keep top k
8. Repeat
```

until stopping criteria are satisfied.

---

# 🧠 Part 13 — Candidate Explosion

Suppose:

```text
beam size = k
vocabulary size = V
```

Each step can conceptually expand:

# `k × V`

candidates.

Example:

```text
k = 5
V = 50,000
```

Potential next candidates:

# `250,000`

before pruning.

In practice, implementations optimize this heavily.

But conceptually:

> wider beams cost more computation.

---

# ⭐ Part 14 — Quality vs Compute Trade-Off

Small beam:

```text
k = 1
```

Fast, but narrow search.

Larger beam:

```text
k = 5
```

Better exploration, more compute.

Very large beam:

```text
k = 100
```

Much more expensive and not necessarily proportionally better.

So:

```text
Larger Beam
→ more search
→ more memory
→ more compute
→ usually more latency
```

---

# 📖 Part 15 — What Happens When `<EOS>` Appears?

Suppose a beam produces:

```text
A B <EOS>
```

That hypothesis is:

# **completed**

It should not be expanded further.

Other active beams may continue.

So Beam Search typically maintains:

* active hypotheses
* completed hypotheses

---

# 🧠 Part 16 — Active vs Completed Beams

Example:

Beam size:

`2`

Current candidates:

```text
A B <EOS> → score -2.1
A C       → score -1.8
```

First candidate:

> finished.

Second:

> still active.

The decoder may continue expanding:

`A C`

while retaining the finished sequence as a possible final answer.

Exact implementation details vary.

---

# 🚨 Part 17 — Do Not Expand After `<EOS>`

Once:

`<EOS>`

is selected:

> that sequence is finished.

You should not generate:

```text
A B <EOS> X Y
```

as if the sequence remained active.

This is a common implementation mistake.

---

# ⭐ Part 18 — When Can Beam Search Stop?

One simple approach:

> continue until every beam is finished or max length is reached.

But optimized implementations can sometimes stop earlier.

For example:

> if no active beam can possibly beat the best finished sequence under the scoring rule.

Exact early-stopping rules depend on:

* scoring
* length normalization
* framework implementation

For interviews, the safe mental model is:

```text
all relevant beams completed
OR
max_length reached
```

---

# 🧠 Part 19 — Why Raw Scores Prefer Short Sequences

Recall:

```text
P(Y|X)
=
product of probabilities
```

Each additional token usually multiplies by:

`< 1`

So:

> longer sequences naturally receive lower raw probability.

In log space:

each token usually adds another:

> negative number.

Therefore:

```text
longer sequence
→ more negative cumulative log score
```

This creates:

# **length bias**

---

# 🧮 Part 20 — Short vs Long Example

Candidate A:

```text
A <EOS>
```

Probabilities:

```text
0.7 × 0.8 = 0.56
```

Candidate B:

```text
B C D <EOS>
```

Probabilities:

```text
0.8 × 0.8 × 0.8 × 0.8
= 0.4096
```

Even though every token in B has high probability:

> the longer sequence gets a lower raw product.

This can encourage:

> overly short outputs.

---

# ⭐ Part 21 — Length Normalization Intuition

To reduce short-sequence bias, we can normalize score by length.

Simple conceptual version:

# `normalized_score = log P(Y|X) / length`

Now we compare:

> average log probability per token

rather than only:

> cumulative log probability.

Exact formulas vary.

Some systems use a length penalty rather than simple division.

---

# 📐 Part 22 — Generic Length Penalty

A common style is:

# `score(Y) = log P(Y|X) / LP(Y)`

where:

`LP(Y)`

is a function of sequence length.

The exact function and hyperparameter can vary.

Important interview point:

> length normalization compensates for the fact that cumulative log probability naturally penalizes longer hypotheses.

---

# 🚨 Part 23 — Length Normalization Is Not Free

If we over-reward long sequences:

> outputs may become unnecessarily long.

If we under-compensate:

> outputs may terminate too early.

So length penalty is:

> a decoding hyperparameter.

It must be tuned for the task.

---

# 🧠 Part 24 — Beam Search Is Not Guaranteed Globally Optimal

This is important.

Beam Search prunes candidates.

Once a path is removed:

> it is gone forever.

A pruned path might theoretically have produced the best complete sequence later.

So Beam Search is:

# **approximate search**

not:

# **exact exhaustive search**

---

# ⭐ Part 25 — Beam Search vs Exhaustive Search

Exhaustive search:

> explores every sequence.

Beam Search:

> explores only the best `k` partial candidates at each step.

Therefore:

```text
Beam Search
→ much cheaper

but

→ can miss the true global optimum
```

---

# 🧠 Part 26 — Beam Size Trade-Off

Increasing beam size generally reduces:

> aggressive pruning.

So:

```text
k larger
→ more candidate diversity survives
```

But:

> larger beam does not guarantee better task quality forever.

At some point:

* quality gains may saturate
* latency increases
* length bias may become more visible
* model probability may not correlate perfectly with human quality

---

# ⭐ Part 27 — Why Bigger Beam Can Sometimes Hurt

This may sound surprising.

If the model itself assigns high probability to:

> short or generic outputs,

a wider search may become even better at finding those high-probability but undesirable sequences.

So:

# **better search of the model ≠ necessarily better real-world output**

This is a senior-level insight.

---

# 🧠 Part 28 — Search Error vs Model Error

This distinction is very useful.

## Search Error

A good sequence has high model probability, but decoding fails to find it.

Beam Search can help.

## Model Error

The model itself gives low probability to the desired sequence.

Beam Search cannot fix that.

So:

```text
Better Search
can fix
Search Error

but not
Model Error
```

---

# 📖 Part 29 — Example of Search Error

Model probabilities support:

```text
B X <EOS>
```

as the best complete sequence.

But greedy chooses:

```text
A
```

at step 1 and never considers B again.

That is:

> search error.

Beam Search may recover B.

---

# 📖 Part 30 — Example of Model Error

Correct translation should start:

`B`

But model predicts:

```text
A → 0.95
B → 0.01
```

Even a reasonable beam may fail to retain B.

The deeper issue is:

> model scoring.

Beam Search cannot magically make B likely.

---

# ⭐ Part 31 — Beam Search Does Not Change Model Probabilities

Important.

Beam Search uses the same:

# `P(y_t | y_<t, X)`

produced by the model.

It changes only:

> how we search candidate sequences.

So:

```text
Model
→ probability distribution

Beam Search
→ search algorithm
```

---

# 🧠 Part 32 — Beam Search and Hidden States

Each beam has its own generated prefix.

Therefore each beam also has its own:

> decoder state.

Example:

```text
Beam 1:
A B
→ decoder state d^(1)

Beam 2:
A C
→ decoder state d^(2)
```

These states must remain separate.

You cannot accidentally share one evolving hidden state across all beam hypotheses.

---

# 🚨 Part 33 — Why States Diverge

Suppose:

```text
Beam 1 history = A B
Beam 2 history = A C
```

Because previous tokens differ:

> their recurrent decoder states differ.

So:

```text
different history
→ different hidden state
→ different next distribution
```

Beam Search must track this correctly.

---

# 🧠 Part 34 — Parent Beam Tracking

After expanding hypotheses, a surviving candidate must remember:

> which previous beam it came from.

Example:

```text
Beam 0: A
Beam 1: B
```

After expansion:

```text
B X
```

might become the best hypothesis.

The system must copy the decoder state associated with:

`B`

not:

`A`.

This bookkeeping is critical in real implementations.

---

# ⭐ Part 35 — Step-by-Step Beam Size 2 Example

Vocabulary:

```text
A
B
C
<EOS>
```

Beam size:

`2`

## Step 1

```text
A → 0.50
B → 0.40
C → 0.10
```

Keep:

```text
A → 0.50
B → 0.40
```

---

## Step 2 — Expand A

```text
A C → 0.50 × 0.30 = 0.15
A <EOS> → 0.50 × 0.40 = 0.20
```

Expand B:

```text
B C → 0.40 × 0.80 = 0.32
B <EOS> → 0.40 × 0.10 = 0.04
```

Candidate pool:

```text
A C      → 0.15
A <EOS>  → 0.20
B C      → 0.32
B <EOS>  → 0.04
```

Top 2:

```text
B C      → 0.32
A <EOS>  → 0.20
```

Now:

* `B C` active
* `A <EOS>` completed

This is Beam Search.

---

# 🧠 Part 36 — Beam Width Does Not Mean Output Length

Very common confusion.

`beam_size = 5`

means:

> keep five hypotheses.

It does **not** mean:

> generate five tokens.

Beam width and sequence length are independent.

---

# 🚫 Part 37 — Beam Search Is Not Top-k Sampling

Both may involve a number `k`, but they are different.

## Beam Search

Keeps:

> `k` partial sequences.

Deterministic under fixed scoring.

## Top-k Sampling

Restricts the next-token distribution to:

> top `k` tokens

and then samples.

So:

```text
Beam k
→ number of sequence hypotheses

Top-k sampling
→ number of allowed token choices
```

---

# 🧠 Part 38 — Beam Search vs Greedy

| Property                  | Greedy    | Beam Search   |
| ------------------------- | --------- | ------------- |
| Active hypotheses         | 1         | k             |
| Search breadth            | Narrow    | Wider         |
| Compute                   | Low       | Higher        |
| Memory                    | Low       | Higher        |
| Local commitment          | Immediate | Delayed       |
| Global optimum guaranteed | No        | No            |
| Beam size 1               | Greedy    | Same behavior |

---

# ⭐ Part 39 — Beam Search vs Sampling

Beam Search tries to find:

> high-scoring sequences under the model.

Sampling tries to generate:

> sequences according to a stochastic distribution.

Beam Search is naturally useful when:

* one accurate answer is preferred
* sequence fidelity matters

Sampling is useful when:

* diversity matters
* many outputs can be valid

---

# 🧠 Part 40 — Why Translation Often Used Beam Search

Machine translation typically wants:

> one high-quality faithful translation.

So historically, Beam Search became common.

Creative generation often values:

> diversity

so sampling may be more appropriate.

Again:

> decoding strategy should match the task.

---

# 📐 Part 41 — Beam Search Complexity Intuition

Suppose:

* beam size = `k`
* vocabulary size = `V`
* output length = `T`

At each step, conceptually consider:

`k × V`

extensions.

Across target length:

> cost grows roughly with beam width and generated length.

Very loosely:

# `O(T × k × V)`

for the candidate-scoring/search portion, apart from model computation and implementation optimizations.

Do not treat this as an exact full-model complexity formula.

---

# 🚨 Part 42 — Why Beam Search Is Slower

Greedy:

```text
1 active sequence
```

Beam size 5:

```text
5 active sequences
```

The decoder effectively evaluates multiple alternative histories.

Therefore:

```text
beam width ↑
→ compute ↑
→ memory ↑
→ latency ↑
```

---

# 🧠 Part 43 — Batch Size vs Beam Size

Do not confuse these.

### Batch Size

Number of independent source examples processed together.

### Beam Size

Number of candidate target hypotheses kept:

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

beam hypotheses during decoding.

---

# ⭐ Part 44 — Memory Cost

Each beam may need:

* token history
* cumulative score
* hidden state
* LSTM cell state if applicable
* parent index
* completion status

So Beam Search uses more memory than greedy decoding.

For recurrent decoders, this can be significant.

---

# 📖 Part 45 — Production Example

Suppose translation API currently uses:

```text
Greedy
P95 latency = 100 ms
BLEU = 28
```

Beam size 5 gives:

```text
P95 latency = 180 ms
BLEU = 30
```

Question:

> Is Beam Search worth it?

There is no purely mathematical answer.

It depends on:

* quality requirement
* latency SLA
* compute cost
* user experience

This is a system-design trade-off.

---

# 🧠 Part 46 — Beam Width Is a Hyperparameter

You should not assume:

```text
beam size 10
>
beam size 5
>
beam size 2
```

for all tasks.

Tune it empirically.

Common evaluation:

```text
beam 1
beam 2
beam 4
beam 8
```

Compare:

* quality
* latency
* memory
* output length
* termination behavior

---

# 🚨 Part 47 — Beware Beam Search and Short Outputs

Without suitable length handling:

> Beam Search may prefer shorter hypotheses.

Because:

```text
each extra token
→ adds another negative log probability
```

This can make:

`<EOS>`

attractive too early.

So always reason about:

> sequence scoring + length.

---

# ⭐ Part 48 — Length Penalty Intuition Example

Suppose:

Candidate A:

```text
length = 2
log score = -1.0
```

Candidate B:

```text
length = 5
log score = -1.5
```

Raw score prefers A because:

`-1.0 > -1.5`

But average scores:

```text
A: -1.0 / 2 = -0.5
B: -1.5 / 5 = -0.3
```

Now B looks better per token.

This demonstrates why:

> length normalization can change ranking.

---

# 🧠 Part 49 — But Average Score Has Its Own Problems

Simple average log-probability may over-reward:

> overly long sequences.

So real systems often use:

> tuned length penalty formulas.

The important mental model is:

# **length scoring is a trade-off**

not:

> “divide by length and problem solved.”

---

# ⭐ Part 50 — Beam Search Can Return N-Best Outputs

Sometimes we may want:

> more than one final sequence.

For example:

```text
Top 3 translations
```

Beam Search naturally maintains multiple candidates.

So a system may return:

# **N-best hypotheses**

for downstream reranking or human review.

---

# 🧠 Part 51 — Reranking

Suppose Beam Search generates:

```text
Candidate 1
Candidate 2
Candidate 3
```

A second model or rule may rerank them using:

* language quality
* task constraints
* domain scores
* business rules

So Beam Search can be part of a larger:

> generate-and-rerank pipeline.

---

# 🔗 Part 52 — Beam Search Does Not Fix Exposure Bias

Exposure bias comes from:

```text
training on ground-truth histories
vs
inference on generated histories
```

Beam Search still generates:

> model-created histories.

It only keeps more than one.

Therefore:

# **Beam Search ≠ exposure bias solution**

---

# 🔗 Part 53 — Beam Search Does Not Fix Context Bottleneck

Likewise:

Context bottleneck concerns:

> encoder-to-decoder source information flow.

Beam Search concerns:

> target sequence search.

Different problems.

---

# 🧠 Part 54 — Problem Taxonomy

```text
Context Bottleneck
→ source access problem

Teacher Forcing
→ training-history strategy

Exposure Bias
→ train/inference mismatch

Autoregressive Decoding
→ feedback generation loop

Greedy Search
→ keep 1 hypothesis

Beam Search
→ keep k hypotheses

Attention
→ dynamic source access
```

Keep these distinct.

---

# 🎤 Part 55 — 30-Second Interview Answer

> **Beam Search is an approximate decoding algorithm for autoregressive sequence models that keeps the top `k` partial hypotheses at every decoding step instead of committing to a single token like greedy search. Each beam is expanded with possible next tokens, cumulative sequence scores—usually log probabilities—are computed, and only the top `k` candidates survive. This reduces search errors caused by early greedy decisions but increases compute and memory. Beam Search is still approximate because pruning can discard a path that later would have become optimal, and practical implementations often need length normalization because cumulative probabilities naturally favor shorter sequences.**

---

# 🎤 Part 56 — Why Is Beam Search Better Than Greedy?

Strong answer:

> **Greedy decoding immediately discards every path except the locally highest-probability one. Beam Search delays that commitment by retaining multiple high-scoring partial sequences, allowing a slightly weaker early token to survive if it later leads to a better full sequence.**

---

# 🎤 Part 57 — Is Beam Search Guaranteed to Find the Best Sequence?

Strong answer:

> **No. Beam Search is approximate because it prunes hypotheses at every step. Once a candidate is removed from the beam, it cannot recover, so the globally highest-probability sequence may still be missed. Exhaustive search would be exact but is computationally infeasible for realistic vocabularies and sequence lengths.**

---

# 🎤 Part 58 — Why Do We Need Length Normalization?

Strong answer:

> **Cumulative sequence probability multiplies probabilities smaller than one, or equivalently accumulates negative log probabilities, so longer sequences naturally receive worse raw scores. This can bias Beam Search toward early `<EOS>` and short outputs. Length normalization or a length penalty adjusts sequence scores to reduce this bias.**

---

# 🚫 Part 59 — High-Yield Traps

### ❌ Beam Search changes model probabilities

No.

---

### ❌ Beam size is output length

No.

---

### ❌ Beam size 1 is different from greedy

Effectively no.

---

### ❌ Beam Search keeps top `k` tokens only

No.

It keeps top `k` partial sequences.

---

### ❌ Beam Search guarantees global optimum

No.

---

### ❌ Larger beam is always better

No.

---

### ❌ Finished `<EOS>` hypotheses should keep expanding

No.

---

### ❌ Raw cumulative score is always fair across lengths

No.

---

### ❌ Length normalization has no trade-offs

No.

---

### ❌ Beam Search fixes model errors

No.

It primarily reduces search errors.

---

### ❌ Beam Search fixes exposure bias

No.

---

### ❌ Beam Search fixes context bottleneck

No.

---

### ❌ Beam Search = top-k sampling

No.

---

# 📐 Formula Flash Card

### Autoregressive Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Log Score

# `S(Y) = Σ_t log P(y_t | y_<t, X)`

### Beam Rule

At each step:

# `keep top k partial sequences by score`

### Generic Length-Normalized Score

# `S_norm(Y) = S(Y) / LP(Y)`

where `LP(Y)` depends on sequence length.

### Beam Size 1

# `Beam(k=1) ≈ Greedy`

---

# 🧠 Part 60 — 10 Things You Must Know Cold

1. **Greedy keeps 1 hypothesis; Beam Search keeps `k`.**
2. **Beam width controls search breadth.**
3. **Beam size 1 is effectively greedy.**
4. **Each beam is expanded independently.**
5. **Candidates are ranked using cumulative sequence scores.**
6. **Log probabilities are normally used.**
7. **`<EOS>` creates completed hypotheses.**
8. **Raw scores can favor short sequences.**
9. **Length normalization/penalty addresses length bias.**
10. **Beam Search reduces search error but does not fix model error.**

---

# 🧠 Final Mental Model

Greedy:

```text
Step 1:
A  ← keep
B  ← discard forever
C  ← discard forever
```

Beam Search:

```text
Step 1:
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

The complete loop is:

```text
Current Beams
↓
Expand Every Beam
↓
Score All Candidates
↓
Rank
↓
Prune to Top k
↓
Separate Finished / Active
↓
Repeat
```

And remember the two key limitations:

```text
Beam Search
↓
better exploration than greedy

but

still approximate
+
more expensive
```

---

# 🚀 Where We Go Next

Beam Search improves:

> how we search through target sequences.

But notice something important.

We have now improved:

* training history with Teacher Forcing
* inference with autoregressive decoding
* target sequence search with Beam Search

Yet our decoder may still suffer from the earlier architectural problem:

```text
Entire Source
↓
One Fixed Context
↓
Decoder
```

We already know that one fixed context becomes restrictive.

The next question is:

> How exactly should the decoder decide which source information is relevant for the current output token?

That is where our sequence-model story starts turning toward one of the biggest breakthroughs in deep learning.

Our next lecture:

# `14_Attention_Motivation`

The progression:

```text
Classic Seq2Seq
↓
Context Bottleneck
↓
Teacher Forcing
↓
Autoregressive Inference
↓
Greedy Search Limitation
↓
Beam Search
↓
Target Search Improved
↓
But Source Access Is Still Fixed
↓
Attention Motivation ← NEXT
```

---

# ⭐ Golden Rule

> **Beam Search does not change what the model believes; it changes how aggressively we search through those beliefs. Instead of committing to one locally best path, it keeps several promising sequence hypotheses alive—but because it still prunes, it remains an approximation with a quality–latency trade-off.**
