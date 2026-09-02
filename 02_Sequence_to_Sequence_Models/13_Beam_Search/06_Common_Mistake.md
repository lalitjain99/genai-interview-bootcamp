# 🚫 Common Mistakes — Beam Search

> **Goal:** Avoid the most common conceptual and implementation mistakes around beam width, candidate expansion, sequence scoring, pruning, `<EOS>`, length bias, decoder-state tracking, and production trade-offs.

---

## ❌ Mistake 1: Thinking Beam Search Keeps the Top `k` Tokens

### Wrong

Beam Search keeps:

> the top `k` next-token choices.

### Correct

Beam Search keeps:

> the top `k` partial sequence hypotheses.

Example:

```text
Beam 1:
A B

Beam 2:
A C

Beam 3:
D E
```

These are:

> complete partial paths,

not just individual tokens.

---

## ❌ Mistake 2: Confusing Beam Size With Output Length

If:

```text
beam_size = 5
```

it means:

> retain 5 hypotheses after pruning.

It does **not** mean:

> generate exactly 5 tokens.

Beam width and output length are independent.

---

## ❌ Mistake 3: Thinking Beam Size 1 Is Very Different From Greedy Search

With:

```text
beam_size = 1
```

only one hypothesis survives after every step.

That is effectively:

# **Greedy Decoding**

So:

```text
Beam(k=1) ≈ Greedy
```

---

## ❌ Mistake 4: Ranking Candidates Only by the New Token Probability

Suppose:

```text
Beam A score = 0.60
next token probability = 0.40
```

and:

```text
Beam B score = 0.30
next token probability = 0.90
```

You should not compare:

```text
0.40 vs 0.90
```

alone.

You must compare cumulative path scores:

```text
A-path = 0.60 × 0.40 = 0.24

B-path = 0.30 × 0.90 = 0.27
```

Beam Search ranks:

> complete partial hypotheses,

not isolated token probabilities.

---

## ❌ Mistake 5: Reinitializing the Sequence Score at Every Step

The score should accumulate.

Correct:

# `S_t = S_(t-1) + log P(y_t | y_<t, X)`

Wrong:

> replace the old beam score with only the current token score.

Beam Search requires:

> cumulative sequence scoring.

---

## ❌ Mistake 6: Using Raw Probability Multiplication for Long Sequences Without Care

Sequence probability is:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

But repeated multiplication can create extremely small numbers.

Practical decoding usually works with:

# `S(Y) = Σ_t log P(y_t | y_<t, X)`

because it is:

* numerically stable
* easy to accumulate

---

## ❌ Mistake 7: Thinking More Negative Log Score Is Better

Log probabilities are usually negative.

Example:

```text
Candidate A → -1.5
Candidate B → -3.2
```

Candidate A is better because:

# `-1.5 > -3.2`

Remember:

> less negative = higher probability = better score.

---

## ❌ Mistake 8: Pruning Separately Inside Each Parent Beam

Suppose beam size is:

`2`

and current beams are:

```text
A
B
```

After expansion:

```text
AX
AY
AZ

BX
BY
BZ
```

You should not simply keep:

```text
best child from A
+
best child from B
```

Beam Search normally pools all expansions:

```text
AX
AY
AZ
BX
BY
BZ
```

then ranks them globally and keeps:

> the top `k`.

A single parent may contribute multiple surviving children.

---

## ❌ Mistake 9: Assuming Every Old Beam Must Survive

No.

Suppose:

```text
Beam A
Beam B
```

expand into candidates.

The next beam might be:

```text
A X
A Y
```

and all children of B may disappear.

Beam Search preserves:

> top-scoring hypotheses,

not parent diversity automatically.

---

## ❌ Mistake 10: Continuing to Expand a Hypothesis After `<EOS>`

If a candidate becomes:

```text
A B <EOS>
```

it is complete.

Wrong:

```text
A B <EOS> C D
```

Correct behavior:

> store it as a completed hypothesis and stop normal expansion of that path.

---

## ❌ Mistake 11: Immediately Returning the First Hypothesis That Emits `<EOS>`

The first completed hypothesis is not necessarily:

> the final best sequence.

Other active beams may later finish with better scores.

So Beam Search typically tracks:

```text
Completed Hypotheses
+
Active Hypotheses
```

and compares them according to the chosen scoring/stopping rule.

---

## ❌ Mistake 12: Assuming Beam Search Always Finds the Global Optimum

Beam Search prunes.

Once a candidate is removed:

```text
Candidate
↓
PRUNED
↓
gone forever
```

That path may theoretically have produced a better complete sequence later.

Therefore Beam Search is:

# **approximate search**

not exhaustive search.

---

## ❌ Mistake 13: Thinking Larger Beam Is Always Better

Increasing beam width provides:

> more exploration.

But also:

* more compute
* more memory
* more latency

And quality may:

> saturate or even degrade.

So:

```text
beam 10
```

is not automatically better than:

```text
beam 4
```

Production beam size should be:

> tuned empirically.

---

## ❌ Mistake 14: Ignoring Length Bias

Consider cumulative log score:

```text
Token 1 → negative value
Token 2 → another negative value
Token 3 → another negative value
```

Longer sequences naturally accumulate:

> more negative scores.

So raw Beam Search scoring can favor:

> short outputs and early `<EOS>`.

This is known as:

# **length bias**

---

## ❌ Mistake 15: Thinking Length Normalization Has One Universal Formula

A simple idea is:

# `score / length`

but that is not universally optimal.

Different implementations may use:

* average log probability
* tuned length penalty
* nonlinear penalty functions

The principle is:

> compensate for sequence-length effects.

The exact formula is a:

> decoding design choice.

---

## ❌ Mistake 16: Assuming Stronger Length Penalty Is Always Better

Too little length compensation:

```text
outputs too short
```

Too much:

```text
outputs too long
```

Length penalty is not:

> a free quality improvement.

It must be tuned.

---

## ❌ Mistake 17: Sharing One RNN Decoder State Across All Beams

Suppose:

```text
Beam 1 = A B
Beam 2 = A C
```

These histories differ.

Therefore their recurrent states differ:

```text
d^(1) != d^(2)
```

Each beam needs its own state.

For LSTM, that includes:

* hidden state `h`
* cell state `c`

Sharing state across beams corrupts decoding.

---

## ❌ Mistake 18: Forgetting Parent-Beam Reordering

Suppose:

```text
Beam 0 = A
Beam 1 = B
```

After expansion, surviving candidates are:

```text
B X
B Y
```

Both came from:

> Beam 1.

Their next recurrent states must therefore originate from:

> Beam 1's state.

Beam Search implementations must correctly track and reorder:

* beam parent IDs
* hidden states
* cell states
* token histories

---

## ❌ Mistake 19: Confusing Batch Size With Beam Size

### Batch Size

Number of independent source inputs.

### Beam Size

Number of target hypotheses:

> per source input.

Example:

```text
batch_size = 8
beam_size = 4
```

may require managing roughly:

```text
8 × 4 = 32
```

hypotheses.

---

## ❌ Mistake 20: Confusing Beam Search With Top-k Sampling

Both contain a `k`, but mean different things.

### Beam Search

```text
k = number of sequence hypotheses kept
```

### Top-k Sampling

```text
k = number of highest-scoring tokens allowed before sampling
```

Beam Search is:

> sequence search.

Top-k sampling is:

> stochastic token selection.

---

## ❌ Mistake 21: Thinking Beam Search Changes the Model's Probabilities

It does not.

The model still produces:

# `P(y_t | y_<t, X)`

Beam Search only decides:

> which candidate histories to continue exploring.

So:

```text
Model
→ scores

Beam Search
→ search
```

---

## ❌ Mistake 22: Expecting Beam Search to Fix Model Error

Suppose the correct token gets:

```text
0.001
```

while a wrong token gets:

```text
0.90
```

The underlying problem is:

> model scoring.

Increasing beam width does not teach the model:

> what the correct sequence should be.

Beam Search mainly helps with:

# **search error**

not:

# **model error**

---

## ❌ Mistake 23: Thinking Beam Search Fixes Exposure Bias

Exposure bias comes from:

```text
Training
→ ground-truth history

Inference
→ generated history
```

Beam Search still operates on:

> generated histories.

It simply keeps more of them.

So:

# `Beam Search ≠ Exposure Bias Solution`

---

## ❌ Mistake 24: Thinking Beam Search Fixes the Context Bottleneck

Context bottleneck concerns:

> how source information reaches the decoder.

Beam Search concerns:

> how target output sequences are explored.

Completely different problems.

```text
Context Bottleneck
→ source access

Beam Search
→ target search
```

---

## ❌ Mistake 25: Assuming a Wider Beam Must Produce More Diverse Outputs

Not necessarily.

Beam Search prioritizes:

> highest-scoring model hypotheses.

A wider beam may actually find more:

* generic
* safe
* similar

high-probability outputs.

Beam width increases:

> search breadth,

not necessarily:

> final-output diversity.

---

## ❌ Mistake 26: Treating Better Model Score as Automatically Better Human Quality

A wider beam may find a sequence with higher:

> model probability.

But humans may prefer another output.

Therefore:

# **Higher Model Score ≠ Guaranteed Better Human Quality**

This is why production evaluation needs:

* task metrics
* human evaluation
* domain constraints
* latency/cost measurements

not just beam scores.

---

# ⚡ Quick Trap Table

| Wrong Idea                                | Correct Idea                    |
| ----------------------------------------- | ------------------------------- |
| Beam keeps top `k` tokens                 | Keeps top `k` partial sequences |
| Beam size = sequence length               | Independent concepts            |
| Beam 1 differs fundamentally from greedy  | Effectively greedy              |
| Rank only current token probability       | Rank cumulative path score      |
| Reset score each timestep                 | Accumulate score                |
| Raw products are ideal                    | Log scores preferred            |
| More negative log score is better         | Less negative is better         |
| Keep best child from every parent         | Pool globally, then top `k`     |
| Every parent must survive                 | Any parent may disappear        |
| Expand after `<EOS>`                      | Mark completed                  |
| First `<EOS>` wins automatically          | Other beams may still win       |
| Beam Search is exact                      | Approximate                     |
| Larger beam always improves output        | Not guaranteed                  |
| Length has no effect on score             | Raw score favors shorter paths  |
| One length-normalization formula fits all | Implementation-dependent        |
| Bigger length penalty always better       | Can over-lengthen outputs       |
| All beams share decoder state             | Each beam needs its own         |
| Parent state tracking isn't important     | Critical                        |
| Beam size = batch size                    | Different dimensions            |
| Beam Search = top-k sampling              | Different algorithms            |
| Beam Search changes model probabilities   | No                              |
| Beam Search fixes model error             | No                              |
| Beam Search fixes exposure bias           | No                              |
| Beam Search fixes context bottleneck      | No                              |

---

# 🧠 Beam Search Debugging Checklist

When debugging Beam Search, verify:

```text
1. Is beam width interpreted as number of hypotheses?
2. Are all active beams expanded?
3. Are candidate scores cumulative?
4. Are log probabilities used consistently?
5. Are candidates ranked globally after expansion?
6. Is pruning actually keeping top k hypotheses?
7. Are completed <EOS> hypotheses handled separately?
8. Are completed hypotheses prevented from normal expansion?
9. Is the stopping criterion implemented correctly?
10. Is max_length still enforced?
11. Is length bias being monitored?
12. Is the length penalty applied consistently?
13. Does every beam retain its own decoder state?
14. Are parent-beam indices correctly tracked?
15. Are hidden/cell states reordered with beam parents?
16. Is batch dimension being confused with beam dimension?
17. Is wider beam actually improving task quality?
18. Is latency increasing proportionally with beam width?
19. Are search errors being confused with model errors?
20. Are outputs being judged only by model score?
```

---

# 📐 Formula Sanity Check

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Cumulative Log Score

# `S(Y) = Σ_t log P(y_t | y_<t, X)`

### Incremental Update

# `S_new = S_parent + log P(next_token | prefix, X)`

### Beam Rule

# `Keep top k candidate hypotheses`

### Generic Length-Adjusted Score

# `S_norm(Y) = S(Y) / LP(Y)`

where:

`LP(Y)`

depends on sequence length.

### Special Case

# `Beam(k=1) ≈ Greedy`

---

# 🧠 Final Mental Model

Do not think Beam Search works like:

```text
Step 1:
choose top k words

Step 2:
choose top k words
```

Think:

```text
Partial Sequence 1
Partial Sequence 2
...
Partial Sequence k
        │
        ▼
Expand each sequence
        │
        ▼
All candidate extensions
        │
        ▼
Add cumulative scores
        │
        ▼
Global ranking
        │
        ▼
Keep top k sequences
        │
        ▼
Repeat
```

And each beam carries its own:

```text
Token History
+
Sequence Score
+
Decoder State
+
Completion Status
```

The central trade-off is:

```text
Beam Width ↑
↓
More Paths Survive
↓
Search Error May Decrease
↓
Compute + Memory + Latency Increase
```

But always remember:

```text
Better Search
≠
Better Model
```

and:

```text
Higher Model Score
≠
Guaranteed Better Human Output
```

---

# ⭐ Golden Rule

> **The biggest Beam Search mistake is to think of it as choosing several good tokens. It is actually maintaining several competing sequence histories, each with its own cumulative score and decoder state, and repeatedly pruning those histories under a scoring rule that itself can introduce trade-offs such as length bias.**
