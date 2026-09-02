# 🚫 Common Mistakes — Inference and Autoregressive Decoding

> **Goal:** Avoid the most common conceptual and implementation mistakes around free-running inference, autoregressive feedback, logits, softmax, token selection, stopping conditions, error propagation, and greedy decoding.

---

## ❌ Mistake 1: Thinking Inference Still Has the Correct Previous Target Token

### Wrong

During inference:

```text id="inf-mistake-1"
predict y1
↓
feed TRUE y1
↓
predict y2
```

### Correct

The ground-truth target is unavailable.

So inference uses:

```text id="inf-mistake-2"
predict ŷ1
↓
feed ŷ1
↓
predict ŷ2
```

This is what makes inference:

> free-running and autoregressive.

---

## ❌ Mistake 2: Thinking the Decoder Hidden State Is the Output Token

The decoder produces:

`d_t`

which is a hidden representation.

It still needs:

```text id="inf-mistake-3"
d_t
↓
Output Projection
↓
Logits
↓
Probability Distribution
↓
Token Selection
```

So:

# `d_t ≠ generated token`

---

## ❌ Mistake 3: Confusing Logits With Probabilities

Logits can be:

```text id="inf-mistake-4"
A → 3.4
B → 0.8
C → -1.2
```

They are arbitrary real-valued scores.

Probabilities require normalization:

# `softmax(logits)`

So:

```text id="inf-mistake-5"
Logits
≠
Probabilities
```

---

## ❌ Mistake 4: Saying Softmax Selects the Next Token

Softmax does not choose anything.

It produces:

> a probability distribution over the vocabulary.

Example:

```text id="inf-mistake-6"
A → 0.60
B → 0.25
C → 0.10
<EOS> → 0.05
```

Then a decoding strategy decides what to select.

For greedy decoding:

# `argmax`

selects the token.

---

## ❌ Mistake 5: Thinking Autoregressive Decoding Generates All Tokens at Once

No.

Generation is sequential:

```text id="inf-mistake-7"
ŷ1
↓
ŷ2
↓
ŷ3
↓
ŷ4
```

because each future token depends on:

> previously generated tokens.

For a recurrent decoder, the hidden state is sequential too.

---

## ❌ Mistake 6: Forgetting to Feed the Selected Token Back

Suppose the model selects:

`A`.

The next decoder step should normally use:

```text id="inf-mistake-8"
A
↓
Embedding
↓
Decoder
```

If you repeatedly feed `<BOS>` instead, the model is no longer performing the intended autoregressive process.

The selected token must become:

> part of the next decoding history.

---

## ❌ Mistake 7: Thinking `<EOS>` and `PAD` Mean the Same Thing

They are completely different.

### `<EOS>`

> The generated sequence has ended.

### `PAD`

> Artificial placeholder used for batching.

Example:

```text id="inf-mistake-9"
A B <EOS> PAD PAD
```

Actual sequence:

```text id="inf-mistake-10"
A B
```

---

## ❌ Mistake 8: Assuming `<EOS>` Makes `max_length` Unnecessary

The model may fail to generate `<EOS>`.

For example:

```text id="inf-mistake-11"
A B B B B B B ...
```

So a production decoder usually has:

```text id="inf-mistake-12"
if token == <EOS>
→ stop

OR

if length >= max_length
→ stop
```

`max_length` is a:

> safety safeguard.

---

## ❌ Mistake 9: Thinking `max_length` Solves Bad Generation

Suppose the model repeats:

```text id="inf-mistake-13"
B B B B B B ...
```

Setting:

`max_length = 50`

only gives:

```text id="inf-mistake-14"
B B B ... until 50
```

It limits damage.

It does not solve:

> why the model is repeating.

---

## ❌ Mistake 10: Thinking One Wrong Token Affects Only One Position

Because predictions are fed back:

```text id="inf-mistake-15"
Wrong Token
↓
Different Next Input
↓
Different Hidden State
↓
Different Probability Distribution
↓
Possible Later Errors
```

So an early mistake can change:

> the entire remaining trajectory.

---

## ❌ Mistake 11: Confusing Exposure Bias With Error Propagation

They are related but distinct.

### Exposure Bias

```text id="inf-mistake-16"
Training
→ ground-truth histories

Inference
→ model-generated histories
```

### Error Propagation

```text id="inf-mistake-17"
wrong generated token
→ changes future decoding
```

So:

# **Exposure bias = mismatch**

# **Error propagation = inference consequence**

---

## ❌ Mistake 12: Thinking Greedy Decoding Finds the Best Complete Sequence

Greedy chooses:

> the highest-probability token right now.

Example:

```text id="inf-mistake-18"
A → 0.55
B → 0.45
```

It picks:

`A`.

But perhaps:

```text id="inf-mistake-19"
A-path total = 0.11
B-path total = 0.405
```

So:

# **best local choice ≠ best full sequence**

This is why Beam Search exists.

---

## ❌ Mistake 13: Comparing Long Sequence Scores Using Raw Probability Without Care

Sequence probability multiplies many numbers below 1:

`P(Y|X) = Π_t p_t`

As sequence length grows:

> raw values become extremely small.

So decoding algorithms commonly use:

# `Σ_t log p_t`

instead.

This improves numerical stability.

---

## ❌ Mistake 14: Thinking a Lower Raw Probability Always Means a Worse Long Sequence

Longer sequences naturally multiply more probabilities.

So raw sequence probability tends to decrease with length.

Example:

```text id="inf-mistake-20"
0.8²
>
0.8¹⁰
```

even though token quality was identical.

This is why sequence-search methods may need:

> length-aware scoring.

We will study that in Beam Search.

---

## ❌ Mistake 15: Treating Early `<EOS>` as a Decoder Loop Bug

Suppose:

```text id="inf-mistake-21"
A <EOS>
```

is generated too early.

If `<EOS>` had the highest selected score:

> the loop is behaving correctly.

The deeper question is:

> why did the model assign `<EOS>` such a high score?

Possible issues include:

* training
* history
* length bias
* calibration
* data

---

## ❌ Mistake 16: Assuming Greedy, Sampling, and Beam Search Change the Model

They do not change the learned weights.

They change:

> how the model's output probabilities are converted into a sequence.

So:

```text id="inf-mistake-22"
Same Model
+
Different Decoding Strategy
=
Different Output Behavior
```

---

## ❌ Mistake 17: Thinking Inference and Evaluation Are the Same

### Inference

> Generate output.

### Evaluation

> Measure how good that output is.

Example:

```text id="inf-mistake-23"
English
↓
Generate French
```

= inference.

```text id="inf-mistake-24"
Generated French
+
Reference French
↓
BLEU
```

= evaluation.

---

# ⚡ Quick Trap Table

| Wrong Idea                                             | Correct Idea                              |
| ------------------------------------------------------ | ----------------------------------------- |
| Inference uses true previous targets                   | Uses generated history                    |
| Hidden state is the token                              | Hidden state must be projected            |
| Logits are probabilities                               | Softmax converts logits                   |
| Softmax selects token                                  | Decoding strategy selects                 |
| All tokens generated simultaneously                    | Autoregressive generation is sequential   |
| Selected token need not be fed back                    | It normally becomes next input            |
| `<EOS>` = `PAD`                                        | Different purposes                        |
| `<EOS>` makes max length unnecessary                   | Max length is a safeguard                 |
| Max length fixes repetition                            | Only stops it eventually                  |
| One error stays local                                  | It can alter future trajectory            |
| Exposure bias = error propagation                      | Related, different                        |
| Greedy finds globally best sequence                    | Locally greedy only                       |
| Raw probabilities are always safe for sequence scoring | Log scores are preferred                  |
| Longer sequence lower probability means worse model    | Length itself lowers raw probability      |
| Early `<EOS>` means loop is broken                     | Model may simply score `<EOS>` too highly |
| Different decoding methods change weights              | They change search/selection              |
| Inference = evaluation                                 | Generate vs measure                       |

---

# 🧠 Debugging Checklist

When debugging autoregressive inference, check:

```text id="inf-mistake-25"
1. Does decoding start with <BOS>/<SOS>?
2. Is the previous selected token embedded correctly?
3. Is the recurrent state carried to the next step?
4. Is source context supplied correctly?
5. Are decoder states projected to vocabulary logits?
6. Are logits being confused with probabilities?
7. Is softmax only producing the distribution?
8. What decoding strategy selects the token?
9. Is the selected token actually fed back?
10. Does <EOS> stop generation?
11. Is max_length enforced as a safeguard?
12. Are <EOS> and PAD handled separately?
13. Are completed sequences masked during batch decoding?
14. Is repetition caused by the feedback loop?
15. Is early <EOS> caused by model scores rather than loop logic?
16. Are sequence scores accumulated in log space?
17. Is greedy being assumed to optimize full-sequence probability?
```

---

# 📐 Formula Sanity Check

### Recurrent Decoder

# `d_t = f_dec(e(ŷ_(t-1)), d_(t-1), c)`

### Output Projection

# `z_t = W_o d_t + b_o`

### Token Distribution

# `P(y_t | y_<t, X) = softmax(z_t)`

### Greedy Selection

# `ŷ_t = argmax_y P(y | y_<t, X)`

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Log Sequence Score

# `log P(Y|X) = Σ_t log P(y_t | y_<t, X)`

### Stop Conditions

```text id="inf-mistake-26"
selected token == <EOS>
OR
generated_length >= max_length
```

---

# 🧠 Final Mental Model

Do not collapse inference into:

```text id="inf-mistake-27"
Decoder
↓
Word
```

Think of the complete chain:

```text id="inf-mistake-28"
Previous Selected Token
        │
        ▼
     Embedding
        │
        ▼
      Decoder
        │
        ▼
   Hidden State
        │
        ▼
 Output Projection
        │
        ▼
      Logits
        │
        ▼
     Softmax
        │
        ▼
 Probability Distribution
        │
        ▼
 Decoding Strategy
        │
        ▼
  Selected Token
        │
        └──────────────┐
                       │
                       ▼
                Next Decoder Step
```

And remember:

```text id="inf-mistake-29"
Selected Token
↓
Future History
↓
Future Decoder State
↓
Future Predictions
```

That is why:

# **a decoding choice is not isolated.**

And greedy's central weakness remains:

# **Best Token Now ≠ Best Sequence Overall**

which leads directly to:

# `13_Beam_Search`

---

# ⭐ Golden Rule

> **The biggest mistake in autoregressive inference is to think the model simply “outputs words.” In reality, every step is a pipeline of scoring, normalization, selection, and feedback—and because the selected token becomes future input, even one decoding decision can redirect the entire sequence.**
