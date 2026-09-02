# 🚫 Common Mistakes — Teacher Forcing

> **Goal:** Avoid the most common conceptual mistakes around teacher forcing, target shifting, exposure bias, autoregressive decoding, scheduled sampling, and evaluation.

---

## ❌ Mistake 1: Teacher Forcing Gives the Current Correct Token

### Wrong

At timestep `t`, the decoder is given:

`y_t`

and asked to predict:

`y_t`.

### Correct

Teacher forcing gives:

# `y_(t-1)`

to predict:

# `y_t`

So:

```text
Previous Correct Token
↓
Predict Next Token
```

Teacher forcing does not reveal the current answer.

---

## ❌ Mistake 2: Teacher Forcing and Target Shifting Are Exactly the Same

They are related, but not identical.

### Target Shifting

Defines the alignment:

```text
Decoder Input:
<SOS> A B C

Target:
A B C <EOS>
```

### Teacher Forcing

Determines whether the decoder actually uses:

> the ground-truth previous token

rather than:

> the model's own previous prediction.

So:

```text
Target Shifting
→ input/label alignment

Teacher Forcing
→ source of decoder history
```

---

## ❌ Mistake 3: Teacher Forcing Removes Autoregression

No.

The model still learns:

# `P(y_t | y_<t, X)`

Each token depends on previous target history.

Teacher forcing only changes:

> whether that previous history comes from ground truth or model predictions during training.

---

## ❌ Mistake 4: Teacher Forcing Is an Inference Strategy

No.

Teacher forcing is primarily:

# **a training strategy**

During inference, the true target sequence is unknown.

So the model must normally use:

```text
its own previous prediction
↓
next decoder step
```

---

## ❌ Mistake 5: Teacher Forcing Allows Future Target Tokens

No.

When predicting:

`y_t`

teacher forcing may use:

`y_1 ... y_(t-1)`

but never:

`y_(t+1), y_(t+2), ...`

Correct:

```text
Predict y3
using y1, y2
```

Wrong:

```text
Predict y3
using y4
```

That would be:

> target leakage.

---

## ❌ Mistake 6: Teacher Forcing Uses a Different Loss Function

Usually false.

Teacher forcing changes:

> decoder input history.

The training objective can still be:

# **next-token cross-entropy**

So:

```text
Teacher Forcing
→ input strategy

Cross Entropy
→ loss
```

---

## ❌ Mistake 7: 100% Teacher Forcing Is Always Best

With:

# `p_TF = 1.0`

training is often:

* stable
* fast
* easy to optimize

But inference never has access to:

> the ground-truth previous token.

So full teacher forcing can create a large:

# **training–inference mismatch**

More teacher forcing is not automatically better.

---

## ❌ Mistake 8: 0% Teacher Forcing Is Always Better Because It Matches Inference

Not necessarily.

Early in training, model predictions may be very poor.

Without teacher forcing:

```text
Wrong Prediction
↓
Wrong Next Input
↓
Different State
↓
More Errors
```

Training can become much harder.

So:

# **more inference-like ≠ easier or better training**

---

## ❌ Mistake 9: Exposure Bias Means the Model Overfits

No.

### Overfitting

The model performs well on training data but poorly on unseen data.

### Exposure Bias

The model trains on:

> ground-truth target histories

but infers on:

> self-generated histories.

These are different problems.

---

## ❌ Mistake 10: Exposure Bias and Error Propagation Are the Same Thing

They are related, but distinct.

### Exposure Bias

The model has limited exposure during training to:

> its own imperfect histories.

### Error Propagation

During inference:

```text
Wrong Token
↓
Wrong/Changed Next History
↓
Different Decoder State
↓
Possible More Errors
```

So:

```text
Exposure Bias
→ mismatch

Error Propagation
→ consequence of autoregressive feedback
```

---

## ❌ Mistake 11: Scheduled Sampling Completely Solves Exposure Bias

No.

Scheduled sampling mixes:

* ground-truth tokens
* model-generated tokens

during training.

Example:

```text
Early
→ more teacher forcing

Later
→ less teacher forcing
```

It may help, but:

* optimization becomes more complex
* discrete sampling remains awkward
* there is no guarantee that exposure bias disappears

So:

> mitigation ≠ complete solution.

---

## ❌ Mistake 12: Teacher Forcing Makes an RNN Decoder Parallel Across Time

No.

Even if all shifted target tokens are known during training:

```text
<SOS> A B C
```

an RNN decoder still computes:

```text
d1 → d2 → d3 → d4
```

because each hidden state depends on the previous hidden state.

Teacher forcing removes uncertainty about the token input.

It does not remove:

> recurrent state dependency.

---

## ❌ Mistake 13: The Model's Wrong Prediction Is Ignored Completely During Teacher Forcing

Not exactly.

Even if the wrong discrete token is not fed into the next step, the prediction still contributes to:

> the loss.

Example:

```text
Correct = A
Predicted distribution gives A low probability
↓
Cross-Entropy Loss
↓
Gradient Update
```

So the model is still penalized for the wrong prediction.

Teacher forcing only prevents that wrong token from necessarily becoming:

> the next decoder input.

---

## ❌ Mistake 14: Ground-Truth Token IDs Receive Gradients

The discrete token ID itself does not receive gradients.

Instead:

```text
Ground-Truth Token ID
↓
Embedding Lookup
↓
Embedding Vector
↓
Decoder
```

The learnable target embedding parameters can receive gradients.

Do not think of integer token IDs as differentiable variables.

---

## ❌ Mistake 15: Backpropagation Works Normally Through `argmax`

No.

Suppose:

```text
logits
↓
argmax
↓
predicted token ID
↓
next input
```

`argmax` is a discrete operation and is generally:

> non-differentiable.

Standard backpropagation cannot smoothly propagate through the selected discrete token.

---

## ❌ Mistake 16: Teacher-Forced Validation Is Enough to Judge Generation Quality

No.

Teacher-forced validation gives the model:

> correct previous target tokens.

Production generation uses:

> model-generated previous tokens.

A model can therefore have:

```text
Excellent teacher-forced token accuracy
```

but:

```text
Poor free-running generation
```

Realistic evaluation must include:

> autoregressive decoding.

---

# ⚡ Quick Trap Table

| Wrong Idea                                  | Correct Idea                     |
| ------------------------------------------- | -------------------------------- |
| Teacher forcing gives current answer        | Gives previous correct token     |
| Teacher forcing = target shifting           | Related but distinct             |
| Teacher forcing removes autoregression      | No                               |
| Teacher forcing is inference                | Training strategy                |
| Future targets can be used                  | No                               |
| Teacher forcing changes the loss            | Usually changes inputs, not loss |
| 100% TF is always best                      | Bigger train/inference mismatch  |
| 0% TF is always best                        | Can make training unstable       |
| Exposure bias = overfitting                 | Different concepts               |
| Exposure bias = error propagation           | Related, not identical           |
| Scheduled sampling fully solves mismatch    | No                               |
| Teacher forcing parallelizes RNN time steps | No                               |
| Wrong predictions don't matter under TF     | They still produce loss          |
| Token IDs get gradients                     | Embeddings do                    |
| Argmax is differentiable                    | No                               |
| Teacher-forced validation is enough         | Free-running evaluation needed   |

---

# 🧠 Debugging Checklist

When reviewing teacher-forced decoder training, ask:

```text
1. Is the target shifted correctly?
2. Is <SOS>/<BOS> used as the first decoder input?
3. Is <EOS> included as a prediction target?
4. At step t, are we predicting y_t from y_(t-1)?
5. Are future targets accidentally visible?
6. What is the teacher forcing ratio?
7. Is sampling per token, sequence, or batch?
8. Is teacher forcing used only during training?
9. Are model predictions fed back during free-running evaluation?
10. Are PAD positions masked from the loss?
11. Is teacher-forced accuracy being confused with generation quality?
12. Is exposure bias being confused with overfitting?
13. Is error propagation being measured separately?
14. Does reduced teacher forcing destabilize training?
15. Is scheduled sampling actually improving free-running metrics?
16. Is argmax being treated incorrectly as differentiable?
```

---

# 📐 Formula Sanity Check

### Seq2Seq Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Full Teacher-Forced Training

# `P(y_t^* | y_<t^*, X)`

### Token Loss

`L_t = -log P(y_t^* | y_<t^*, X)`

### Total Loss

# `L = -Σ_t log P(y_t^* | y_<t^*, X)`

### Teacher Forcing Ratio

# `p_TF`

```text
with probability p_TF
→ use ground-truth previous token

with probability 1 - p_TF
→ use model-generated previous token
```

---

# 🧠 Final Mental Model

Do not think:

```text
Teacher Forcing
=
Give Model the Answer
```

Think:

```text
Teacher Forcing
=
Give Model the Correct History
```

Training:

```text
<SOS>
↓
predict y1
↓
feed TRUE y1
↓
predict y2
↓
feed TRUE y2
```

Inference:

```text
<SOS>
↓
predict ŷ1
↓
feed ŷ1
↓
predict ŷ2
```

This creates:

```text
Training History
≠
Inference History
↓
Exposure Bias
```

And once inference makes an error:

```text
Wrong Token
↓
Changed History
↓
Changed Hidden State
↓
Potential Error Cascade
```

The central trade-off:

```text
High Teacher Forcing
→ easier learning
→ larger mismatch

Low Teacher Forcing
→ harder learning
→ more inference-like experience
```

---

# ⭐ Golden Rule

> **The biggest mistake is to think teacher forcing makes the prediction task easier by revealing the answer. It does not reveal the current answer—it only gives the decoder the correct previous history, which stabilizes training but creates a mismatch when that teacher is absent during inference.**
