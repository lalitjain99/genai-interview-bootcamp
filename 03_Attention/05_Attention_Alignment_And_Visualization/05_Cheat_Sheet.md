# ⚡ Attention Alignment and Visualization — Cheat Sheet

> Goal: Reconstruct attention alignment, matrix interpretation, heatmap patterns, coverage, debugging use cases, and explainability limitations in a few minutes before an interview.

---

# 1. 🎯 One-Line Definition

> **Attention alignment is the collection of attention weights across all decoder timesteps, usually visualized as a source-target heatmap.**

---

# 2. 🧠 Core Idea

At one decoder timestep:

```text
α_t
=
[
α_(t,1),
α_(t,2),
...
α_(t,T_src)
]
```

This is one attention distribution over the source.

Across all target timesteps:

```text
α_1
α_2
...
α_(T_tgt)
```

Stack them:

```text
↓
Alignment Matrix
```

---

# 3. 📐 Alignment Matrix

Define:

```text
A_(t,i)
=
α_(t,i)
```

Then:

```text
A ∈ R^(T_tgt × T_src)
```

Using our convention:

```text
rows
→ target / decoder timesteps

columns
→ source / encoder positions
```

---

# 4. ⭐ What Does One Cell Mean?

```text
A[t,i]
=
α_(t,i)
```

means:

> how much encoder representation `h_i` is weighted while constructing the context for target timestep `t`.

---

# 5. 🚨 Heatmap Orientation Is Not Universal

Some plots use:

```text
rows = target
columns = source
```

Others may transpose them.

So:

> **Always read the axis labels.**

Do not memorize one plotting orientation as universal.

---

# 6. 📐 Row Normalization

For each target timestep:

```text
Σ_i α_(t,i) = 1
```

Therefore:

```text
each row normally sums to 1
```

under standard source-wise softmax.

---

# 7. 🚫 Columns Do Not Need to Sum to 1

A source position can receive attention from many target timesteps.

So generally:

```text
Σ_t α_(t,i) ≠ 1
```

This quantity is closer to a:

> coverage-style measure

than a probability distribution.

---

# 8. 🧮 Shape Example

If:

```text
T_src = 8
T_tgt = 5
```

then:

```text
A ∈ R^(5 × 8)
```

and total attention entries are:

```text
5 × 8 = 40
```

---

# 9. 📖 Heatmap

The alignment matrix can be visualized as a heatmap.

Conceptually:

```text
small attention weight
→ light cell

large attention weight
→ dark cell
```

This makes patterns visually obvious.

---

# 10. ⭐ Diagonal Alignment

Example:

```text
        x1    x2    x3

y1     0.9   0.05  0.05
y2     0.05  0.9   0.05
y3     0.05  0.05  0.9
```

This often suggests:

```text
source order
≈
target order
```

or approximately monotonic alignment.

---

# 11. 🔄 Reordering

Example:

```text
Source:
A B C

Target:
A C B
```

Attention may align:

```text
A' → A
C' → C
B' → B
```

So the strongest heatmap values may jump around.

Important:

> **Non-diagonal attention can be completely correct.**

---

# 12. 🧠 Attention Is Usually Soft

Hard alignment:

```text
[0, 0, 1, 0]
```

means:

> choose one source position.

Soft alignment:

```text
[0.05, 0.15, 0.65, 0.15]
```

means:

> multiple source positions contribute.

---

# 13. ⭐ Phrase-Level Alignment

Attention is not limited to word-to-word mappings.

A target token may depend on:

```text
source position 3
+
source position 4
```

For example:

```text
[0.02, 0.03, 0.30, 0.65]
```

So attention can model:

* one-to-one
* many-to-one
* one-to-many
* many-to-many

soft relationships.

---

# 14. 🎯 Sharp Attention

Example:

```text
[0.01, 0.02, 0.94, 0.03]
```

Interpretation:

> most weight is concentrated on one source representation.

Often useful for highly localized information such as:

* names
* dates
* numbers

But:

# **Sharp ≠ automatically correct**

---

# 15. 🌫️ Diffuse Attention

Example:

```text
[0.20, 0.25, 0.30, 0.25]
```

Interpretation:

> multiple source positions contribute meaningfully.

This can be valid for:

* phrases
* grammar
* sentence-level context
* distributed evidence

So:

# **Diffuse ≠ automatically bad**

---

# 16. 📐 Attention Entropy

Concentration can be described using:

```text
H(α_t)
=
- Σ_i α_(t,i) log α_(t,i)
```

### Low entropy

```text
[0.98, 0.01, 0.01]
```

→ concentrated / sharp

### High entropy

```text
[0.34, 0.33, 0.33]
```

→ diffuse

Important:

> lower entropy is not inherently better.

---

# 17. 🔁 Repeated Attention

Suppose:

```text
t1 → source 4 = 0.80
t2 → source 4 = 0.78
t3 → source 4 = 0.84
```

Possible interpretations:

* same source representation supports multiple target tokens;
* repeated generation;
* poor coverage;
* decoder stuck around one source region.

The heatmap provides:

> a debugging clue, not proof.

---

# 18. 👀 Ignored Source Regions

If an important source position receives:

```text
almost zero attention
```

throughout decoding, and the corresponding information is missing from output, that may suggest:

> insufficient source retrieval.

Still inspect:

* encoder representation
* decoder logits
* decoding strategy
* training data
* masking

before concluding root cause.

---

# 19. 🛡️ Padding Check

Source:

```text
A B C PAD PAD
```

Correct masking should produce:

```text
α_PAD ≈ 0
```

So PAD columns should appear almost empty.

Significant PAD attention is a strong warning to inspect:

> attention masking.

---

# 20. 📐 Coverage

A simple coverage-style quantity:

```text
coverage_i
=
Σ_t α_(t,i)
```

Interpretation:

> total attention received by source position `i` across target timesteps.

---

# 21. 🧠 Coverage Interpretation

If:

```text
coverage_i ≈ 0
```

the source position was rarely attended to.

If coverage is high:

> the source position may have been repeatedly used.

Coverage-style analysis is useful for studying:

* omission
* repetition

---

# 22. 🚨 Encoder Position ≠ Raw Token Only

Attention weights apply to:

```text
h_i
```

not directly to a raw token.

With a BiRNN/BiLSTM:

```text
h_i
=
[→h_i ; ←h_i]
```

so `h_i` may contain:

```text
token i
+
left context
+
right context
```

Therefore:

> “attention on token i” is shorthand for attention on the contextual encoder representation at position `i`.

---

# 23. ⭐ Contribution vs Causation

High attention means:

> an encoder representation receives a large coefficient in the context vector.

It does **not** prove:

> that representation completely caused the final prediction.

The final output also depends on:

```text
decoder state
+
target history
+
context vector
+
output layers
+
other transformations
```

So:

# **High attention ≠ complete causal explanation**

---

# 24. 🔬 What Gives Stronger Causal Evidence?

Use interventions such as:

* perturbation
* ablation
* masking
* counterfactual inputs
* representation intervention

A useful mental distinction:

```text
Heatmap
→ observation

Ablation / perturbation
→ intervention
```

---

# 25. ⭐ Best Explainability Wording

Avoid:

> “Attention tells us exactly why the model predicted this.”

Prefer:

> **Attention weights show how encoder representations were weighted while constructing the context vector, so they provide useful alignment and diagnostic information, but not a complete causal explanation of the final prediction.**

---

# 26. 🧠 Why Heatmaps Are Still Useful

Attention maps are useful for:

```text
alignment analysis
+
reordering inspection
+
mask debugging
+
coverage analysis
+
error investigation
+
training behavior
```

Think:

# **Diagnostic signal, not final proof**

---

# 27. 🔗 Bahdanau vs Luong Visualization

Bahdanau:

```text
additive score
↓
softmax
↓
α_(t,i)
```

Luong:

```text
multiplicative score
↓
softmax
↓
α_(t,i)
```

Once we have:

```text
α_(t,i)
```

the alignment/visualization concept is the same.

---

# 28. 🚫 Heatmap Does Not Reveal the Scorer

From the final alignment matrix alone, you generally cannot tell whether scores came from:

* Bahdanau
* Luong Dot
* Luong General

because the heatmap contains:

> normalized attention weights

not the internal scoring architecture.

---

# 29. 🌍 Global Attention

Global attention considers:

```text
all source positions
```

at each decoder step.

This produces a full conceptual alignment matrix:

```text
T_tgt × T_src
```

and approximately:

```text
O(T_tgt × T_src)
```

source-target score interactions.

---

# 30. 📍 Local Attention

Local attention considers only:

```text
a smaller source window
```

per decoder timestep.

Heatmaps may show:

> narrow moving bands

rather than full-source access.

Trade-off:

```text
less compute
↔
possible missed information
```

---

# 31. 🧠 Training-Time Pattern

Early training may show:

```text
noisy
diffuse
unstructured
```

attention.

Later training may show:

```text
more task-relevant structure
```

But:

# **Pretty heatmap ≠ better model**

Task performance remains the real evaluation target.

---

# 32. 🚨 Uniform Attention

Example:

```text
[0.25, 0.25, 0.25, 0.25]
```

Persistent uniform attention may suggest:

* weak differentiation between source positions;
* early training;
* scoring issue;
* task genuinely requiring broad context.

It deserves investigation, not automatic rejection.

---

# 33. 🚨 Attention Collapse

Example:

```text
t1 → [0.02, 0.95, 0.02, 0.01]
t2 → [0.01, 0.96, 0.02, 0.01]
t3 → [0.02, 0.94, 0.03, 0.01]
```

Almost every target step focuses on the same source position.

This may suggest:

> attention collapse or repeated-source usage.

Again, interpret together with output behavior.

---

# 34. ⚙️ Debugging Workflow

Best mental flow:

```text
Bad Output
↓
Inspect Heatmap
↓
Observe Pattern
↓
Form Hypothesis
↓
Inspect Mask / Encoder / Decoder / Data
↓
Run Intervention / Test
↓
Confirm or Reject Hypothesis
```

Shortest version:

# **Observe → Hypothesize → Test**

---

# ⚡ 10 Things to Know Cold

1. One decoder step gives one attention distribution.
2. Stack all distributions to get the alignment matrix.
3. Typical shape is `T_tgt × T_src`.
4. Rows usually sum to 1.
5. Columns do not have to sum to 1.
6. Diagonal patterns may indicate similar source-target ordering.
7. Off-diagonal patterns can represent valid reordering.
8. Sharp and diffuse attention can both be correct.
9. Heatmaps are valuable debugging and alignment tools.
10. Attention weights are not complete causal explanations.

---

# 🚫 High-Yield Traps

### ❌ Alignment means exact word-to-word matching

No.

Soft attention can spread weight across several positions.

---

### ❌ Every column must sum to 1

No.

---

### ❌ Diagonal attention is always better

No.

Valid reordering can create non-diagonal patterns.

---

### ❌ Sharp means correct

No.

---

### ❌ Diffuse means confused

No.

---

### ❌ High attention proves causation

No.

---

### ❌ Heatmap orientation is fixed

No.

Check labels.

---

### ❌ `h_i` is just raw token `i`

No.

It may be contextual.

---

# 📐 Formula Card

### Weight

```text
α_(t,i)
=
softmax_i(e_(t,i))
```

### Context

```text
c_t
=
Σ_i α_(t,i) h_i
```

### Alignment Matrix

```text
A_(t,i)
=
α_(t,i)
```

### Shape

```text
A
∈
R^(T_tgt × T_src)
```

### Row Sum

```text
Σ_i A_(t,i)
=
1
```

### Coverage

```text
coverage_i
=
Σ_t α_(t,i)
```

### Entropy

```text
H(α_t)
=
-Σ_i α_(t,i) log α_(t,i)
```

---

# 🎤 30-Second Interview Answer

> **Attention alignment is formed by stacking the attention distribution from each decoder timestep into a `T_tgt × T_src` matrix. Each row shows how one target step distributes attention across source positions. Visualizing this matrix as a heatmap helps inspect alignment, reordering, concentration, repeated focus, ignored regions, masking issues, and coverage. However, attention weights show how encoder representations contribute to the context vector and should not automatically be treated as complete causal explanations of the final prediction.**

---

# 🎤 Strong Short Answer: What Does a Heatmap Show?

> **An attention heatmap shows the normalized source-attention distribution for each target timestep, making it easier to inspect where the decoder retrieves information and how that behavior changes across generation.**

---

# 🎤 Strong Short Answer: Can Attention Explain the Model?

> **It can provide useful alignment and diagnostic evidence, but not a complete causal explanation. For causal claims, I would combine attention visualization with interventions such as perturbation or ablation.**

---

# 🧠 Final Mental Model

```text
One Decoder Step
↓
One Attention Distribution

All Decoder Steps
↓
Stack Distributions

↓
Alignment Matrix

↓
Heatmap

↓
Inspect Patterns

↓
Alignment / Reordering / Coverage / Bugs

↓
Form Hypothesis

↓
Test
```

Shortest version:

# **Weights → Matrix → Heatmap → Insight**

with the critical rule:

# **Observe → Hypothesize → Test**

and never:

# **High attention → automatic causation**

---

# 🔗 Next Topic

We now understand:

```text
how attention is computed
+
how attention creates alignment
+
how alignment can be visualized
```

The final chapter of this classical attention module asks:

> **What important problems still remain even after attention solves the fixed-context bottleneck?**

That leads to:

# **06_Attention_Limitations**

and from those limitations, the motivation for:

# **Transformers**
