# 📝 Attention Alignment and Visualization — Revision Notes

> Goal: Reconstruct how attention weights across decoder timesteps form an alignment matrix, how to read the resulting heatmap, and what conclusions are safe to draw from it.

---

# 🎯 One-Line Definition

> **Attention alignment is the collection of attention distributions across all decoder timesteps, typically visualized as a source-target heatmap.**

---

# 🧠 Core Idea

At one decoder step:

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

This is one distribution over source positions.

Across all target steps:

```text
α_1
α_2
...
α_(T_tgt)
```

Stack them together:

```text
↓
Attention Alignment Matrix
```

---

# 📐 Alignment Matrix

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

So:

```text
A[t,i]
```

means:

> attention assigned to source position `i` at target timestep `t`.

---

# ⭐ Orientation Rule

Do not blindly memorize:

```text
x-axis = source
y-axis = target
```

Some libraries transpose the visualization.

Always inspect the axis labels.

The underlying quantity is still:

```text
α_(t,i)
```

---

# 📐 Row Normalization

For each target timestep:

```text
Σ_i α_(t,i) = 1
```

So:

> each row normally sums to 1.

But:

```text
column sums
```

do not generally equal 1.

A source position can receive attention across many target steps.

---

# 🧮 Small Example

Suppose:

```text
Source:
I     love     AI
```

Target:

```text
Je    aime     IA
```

Attention matrix:

```text
        I      love     AI

Je     0.85    0.10    0.05

aime   0.05    0.90    0.05

IA     0.05    0.10    0.85
```

Each row says:

> where that target step retrieves source information.

---

# 🧠 Diagonal Alignment

A diagonal-looking pattern often means:

```text
source order
≈
target order
```

Example:

```text
source 1 → target 1
source 2 → target 2
source 3 → target 3
```

This often appears when source and target have similar ordering.

---

# 🔄 Reordering

Attention does not require monotonic movement.

Example:

```text
Source:
A B C

Target:
A C B
```

Possible alignment:

```text
A' → A
C' → C
B' → B
```

So the heatmap may move:

```text
forward
then backward
```

This reflects:

> dynamic source access.

---

# 🧠 Attention Is Usually Soft Alignment

Hard alignment:

```text
[0, 0, 1, 0]
```

means:

> choose exactly one source position.

Soft attention:

```text
[0.05, 0.15, 0.65, 0.15]
```

means:

> several source positions can contribute.

Therefore attention can represent:

* one-to-one alignment
* many-to-one alignment
* one-to-many alignment
* phrase-level alignment

---

# ⭐ Phrase-Level Alignment

Suppose one target token depends on:

```text
deep
+
learning
```

Then attention might be:

```text
[0.02, 0.03, 0.30, 0.65]
```

This means the target step retrieves information from multiple source representations.

So:

> attention alignment is not necessarily word-to-word.

---

# 📖 Heatmap

An attention matrix can be visualized as a heatmap.

Conceptually:

```text
small weight
→ light cell

large weight
→ dark cell
```

This makes patterns easier to inspect.

---

# 🧠 What Can a Heatmap Reveal?

A heatmap can help identify:

* source-target alignment
* reordering
* repeated focus
* ignored regions
* sharp attention
* diffuse attention
* padding-mask problems
* long-sequence behavior

---

# 🎯 Sharp Attention

Example:

```text
α_t
=
[0.01, 0.02, 0.94, 0.03]
```

Interpretation:

> most attention is concentrated on one source representation.

Possible cases:

* name
* date
* number
* highly specific content word

But:

> sharp does not automatically mean correct.

---

# 🌫️ Diffuse Attention

Example:

```text
α_t
=
[0.20, 0.25, 0.30, 0.25]
```

Interpretation:

> several source positions contribute.

This may be reasonable when the prediction depends on:

* a phrase
* sentence structure
* broad contextual information

So:

```text
sharp = good
diffuse = bad
```

is an incorrect rule.

---

# 🧠 Attention Entropy

One way to describe concentration is:

```text
H(α_t)
=
- Σ_i α_(t,i) log α_(t,i)
```

### Low entropy

```text
[0.98, 0.01, 0.01]
```

means:

> concentrated attention.

### High entropy

```text
[0.34, 0.33, 0.33]
```

means:

> diffuse attention.

Neither is automatically better.

---

# 🔁 Repeated Attention

Suppose:

```text
t1 → source 4 = 0.80
t2 → source 4 = 0.75
t3 → source 4 = 0.82
```

This may mean:

> one source representation supports multiple target tokens.

That can be valid.

But excessive repeated attention can also be associated with:

* repeated generation
* poor source coverage
* stuck decoding

So investigate in context.

---

# 🧠 Ignored Source Positions

If an important source position receives:

```text
almost zero attention
```

throughout decoding, and that information is omitted from the output, this can be an important debugging signal.

Possible hypothesis:

> the decoder failed to retrieve that source representation.

But the attention map alone does not prove the entire cause.

---

# 🛡️ Padding

Suppose:

```text
Source:
A B C PAD PAD
```

Correct masking should give:

```text
α_PAD ≈ 0
```

So PAD columns should appear nearly empty in the heatmap.

If padding receives significant attention:

> check the attention mask.

---

# 🧠 Special Tokens

Special tokens such as:

```text
<BOS>
<EOS>
<PAD>
```

may appear differently depending on the implementation.

Always check whether they are included in the plotted axes.

Do not assume `<EOS>` must strongly align to one lexical source word.

---

# 📐 Coverage

A rough source-coverage quantity can be defined as:

```text
coverage_i
=
Σ_t α_(t,i)
```

This measures:

> total attention received by source position `i` across target timesteps.

Important:

```text
coverage_i
```

does not have to equal 1.

---

# 🧠 Coverage Intuition

Low coverage:

```text
≈ 0
```

may suggest the source position was rarely attended to.

High coverage may suggest:

> repeated use of the source position.

Coverage-style reasoning has been useful in sequence-generation systems for studying:

* omission
* repetition

---

# 🚨 Encoder State Is Not the Raw Token

When we say:

> target step attends to source token `i`

more precisely, attention operates on:

```text
h_i
```

the encoder representation at source position `i`.

Especially with a BiRNN / BiLSTM:

```text
h_i
=
[→h_i ; ←h_i]
```

the representation contains contextual information from surrounding positions.

So:

```text
high α_(t,i)
```

does not necessarily mean:

> only raw token `i` mattered.

---

# ⭐ Contribution vs Causation

A high attention weight directly tells us:

> `h_i` received a large coefficient when constructing `c_t`.

It does **not** completely tell us:

> why the final token was produced.

The final prediction also depends on:

```text
decoder state
+
target history
+
context vector
+
output layers
+
other learned transformations
```

Therefore:

# **High attention ≠ complete causal explanation**

---

# 🧠 Safer Interpretation

Avoid:

> “The model predicted this because it attended to that word.”

Better:

> “The encoder representation at that source position received a high weight when the model constructed the context for this decoder step.”

This is technically stronger.

---

# 🔬 Causal Analysis Requires More

To investigate causal importance, stronger techniques include:

* ablation
* perturbation
* counterfactual testing
* representation intervention

Attention visualization alone is observational.

---

# ⭐ Why Attention Visualization Is Still Useful

The limitation above does not make attention maps useless.

They remain valuable for:

```text
alignment analysis
+
error analysis
+
mask debugging
+
sequence-behavior inspection
```

A good mental model:

> **Attention maps are diagnostic signals, not final explanations.**

---

# 🔗 Bahdanau vs Luong Visualization

Both ultimately produce:

```text
α_(t,i)
```

So both can produce the same kind of:

```text
alignment heatmap
```

The difference is only how the raw score was generated.

### Bahdanau

```text
additive scorer
```

### Luong

```text
multiplicative scorer
```

Once softmax gives `α`, the visualization concept is the same.

---

# 🚨 Heatmap Alone Does Not Reveal the Scoring Function

If you only see:

```text
A
```

the final normalized attention matrix, you generally cannot determine whether it was produced by:

* Bahdanau
* Luong Dot
* Luong General

because the internal scoring architecture is not directly visible in the final weights.

---

# 🧠 Training Evolution

Early training:

```text
noisy / diffuse / unstructured attention
```

Later training may show:

```text
more task-relevant structure
```

But:

> prettier alignment is not automatically better model quality.

Always evaluate the actual task output.

---

# ⚠️ Uniform Attention

Persistent:

```text
[0.25, 0.25, 0.25, 0.25]
```

means:

> the model is not strongly differentiating source positions through attention.

Possible reasons include:

* early training
* weak scoring function
* task does not require localized access
* model issue

It is worth investigating, not automatically declaring incorrect.

---

# ⚠️ Attention Collapse

Example:

```text
t1 → [0.02, 0.95, 0.02, 0.01]
t2 → [0.01, 0.96, 0.02, 0.01]
t3 → [0.02, 0.94, 0.03, 0.01]
```

Every target step focuses on essentially the same source position.

This may indicate:

> attention collapse

or it may be valid for the task.

Interpret it together with output behavior.

---

# 🧠 Debugging Workflow

A useful workflow is:

```text
Bad Output
↓
Inspect Attention Heatmap
↓
Observe Suspicious Pattern
↓
Create Hypothesis
↓
Inspect Mask / Encoder / Decoder / Data
↓
Test the Hypothesis
```

The heatmap helps:

> generate a hypothesis.

It should not be treated as final proof.

---

# 📐 Matrix Size and Computational Connection

If:

```text
T_src = source length

T_tgt = target length
```

then:

```text
A ∈ R^(T_tgt × T_src)
```

and contains:

```text
T_tgt × T_src
```

attention weights.

This corresponds to the approximate pairwise interaction structure of global classical attention:

```text
O(T_src × T_tgt)
```

---

# 🌍 Global vs Local Visualization

## Global Attention

Each target step can consider:

```text
all source positions
```

so the full alignment matrix is available.

## Local Attention

Each target step considers only:

```text
a smaller source window
```

The heatmap may therefore show narrow moving bands.

Trade-off:

```text
Global
→ more accessibility
→ more compute

Local
→ less compute
→ possible missed information
```

---

# 🎤 30-Second Interview Answer

> **Attention alignment is obtained by stacking the attention distribution from every decoder timestep into a source-target matrix. If the target length is `T_tgt` and source length is `T_src`, the matrix is typically `T_tgt × T_src`. Each row shows how one target step distributes attention across source positions. Visualizing this as a heatmap helps inspect alignment, reordering, concentration, masking, and source coverage. However, attention weights describe how encoder states contribute to the context vector and should not automatically be treated as complete causal explanations of the prediction.**

---

# 🎤 What Does One Cell Mean?

```text
A[t,i]
=
α_(t,i)
```

means:

> how much encoder representation `h_i` contributes through attention when building the context for target step `t`.

---

# 🎤 Do Rows and Columns Both Sum to 1?

No.

Typically:

```text
row sum = 1
```

because softmax is applied across source positions for each target step.

But:

```text
column sum
```

is unconstrained.

---

# 🎤 What Does a Diagonal Pattern Mean?

> It often indicates that source and target ordering are approximately monotonic or similar.

It does not mean every translation must show a diagonal.

---

# 🎤 Can Attention Explain the Prediction?

Strong answer:

> **Attention weights are useful for understanding alignment and how encoder representations are weighted in the context vector, but they are not sufficient by themselves to establish complete causal importance for the final prediction.**

---

# 🚨 High-Yield Mistakes

### ❌ Alignment means one-to-one mapping

No.

Attention is usually soft.

---

### ❌ Every column sums to 1

No.

Rows usually do.

---

### ❌ Diagonal attention is always best

No.

Reordering can be correct.

---

### ❌ Sharp attention is always better

No.

---

### ❌ Diffuse attention means failure

No.

---

### ❌ Heatmap axes are universally fixed

No.

Check labels.

---

### ❌ High attention proves causation

No.

---

### ❌ Source position means raw token only

No.

It represents an encoder state, which may be contextual.

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

# ⚡ 10 Things to Know Cold

1. One decoder step produces one attention distribution.
2. Stack all distributions to get the alignment matrix.
3. Matrix shape is usually `T_tgt × T_src`.
4. Rows represent target timesteps.
5. Columns represent source positions.
6. Each row normally sums to 1.
7. Diagonal patterns may indicate similar source-target order.
8. Reordered heatmaps can reflect valid linguistic reordering.
9. Attention maps are useful for debugging and alignment analysis.
10. Attention weights are not complete causal explanations.

---

# 🧠 Final Mental Model

```text
One Decoder Step
↓
One Attention Distribution

Repeat Over Target Steps
↓
Many Attention Distributions

Stack Them
↓
Alignment Matrix

Visualize It
↓
Heatmap

Inspect It
↓
Alignment / Debugging Insight

But Remember
↓
Visualization ≠ Complete Causal Explanation
```

Shortest version:

# **Weights → Matrix → Heatmap → Insight**

with one critical warning:

# **Alignment signal ≠ causation**

---

# 🔗 Next Topic

Attention now gives us:

```text
dynamic source access
+
soft alignment
+
useful visualization
```

But several architectural limitations still remain.

The next question is:

> **What problems are still unsolved even after classical attention?**

That leads to:

# **06_Attention_Limitations**
