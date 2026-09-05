# 📘 Attention Alignment and Visualization

## 🌟 Introduction

We now understand how classical attention computes relevance.

Generic attention gave us:

```text
e_(t,i)
=
score(decoder_state, h_i)
```

Bahdanau showed us one way to compute the score:

```text
e_(t,i)
=
v_a^T tanh(
W_s s_(t-1)
+
W_h h_i
+
b_a
)
```

Luong showed us another:

```text
e_(t,i)
=
s_t^T h_i
```

or:

```text
e_(t,i)
=
s_t^T W_a h_i
```

In both cases, the next steps were the same:

```text
Scores
↓
Softmax
↓
Attention Weights α_(t,i)
↓
Weighted Context c_t
```

Until now, however, we have mostly looked at attention for:

> **one decoder timestep.**

But a complete generated sequence contains many decoder timesteps.

At every timestep, we get a new attention distribution.

So a natural question arises:

> **What happens if we collect all those attention distributions together?**

We obtain something extremely useful:

# **an Attention Alignment Matrix**

And once we have that matrix, we can visualize it.

That visualization helps us understand:

* where the model is looking;
* how source and target positions align;
* whether word reordering occurs;
* whether attention is sharp or diffuse;
* whether padding is being attended to;
* whether some source information is repeatedly ignored.

But it also introduces an important warning:

> **Attention visualization is useful evidence about model behavior, but it is not automatically a causal explanation of the prediction.**

This chapter develops both sides of that idea.

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to:

* explain what source-target alignment means;
* derive an attention alignment matrix from `α_(t,i)`;
* understand the matrix dimensions;
* interpret rows and columns correctly;
* explain diagonal, reordered, sharp, and diffuse attention patterns;
* understand why attention can align phrases rather than individual words;
* reason about repeated and skipped attention;
* explain how padding appears in an attention map;
* understand why soft alignment differs from hard alignment;
* explain how attention heatmaps are produced;
* use attention visualization for debugging;
* understand the limitations of interpreting attention weights;
* distinguish correlation/contribution from causal explanation;
* connect classical alignment visualization to later attention architectures.

---

# 📖 Part 1 — Attention at One Decoder Step

Suppose the source sentence contains four positions:

```text
x1  x2  x3  x4
```

At target timestep:

```text
t = 1
```

attention might produce:

```text
α_1
=
[0.70, 0.20, 0.05, 0.05]
```

At:

```text
t = 2
```

it may produce:

```text
α_2
=
[0.10, 0.70, 0.15, 0.05]
```

At:

```text
t = 3
```

it may produce:

```text
α_3
=
[0.05, 0.10, 0.20, 0.65]
```

Each vector answers:

> Which source positions matter for this target step?

---

# 🧠 Part 2 — One Row per Decoder Step

For every decoder timestep:

```text
t
```

we have:

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

This is one distribution over the source sequence.

So if the target sequence contains:

```text
T_tgt
```

positions, we obtain:

```text
T_tgt
```

attention distributions.

What if we stack them?

---

# 📐 Part 3 — The Alignment Matrix

Stacking all target-step attention distributions gives:

```text
A =
[
 α_(1,1)  α_(1,2)  ... α_(1,T_src)
 α_(2,1)  α_(2,2)  ... α_(2,T_src)
 ...
 α_(T_tgt,1) ...       α_(T_tgt,T_src)
]
```

Therefore:

```text
A ∈ R^(T_tgt × T_src)
```

where:

```text
rows
→ target / decoder positions

columns
→ source / encoder positions
```

This matrix is often called:

> **attention matrix**

or conceptually:

> **alignment matrix**

---

# ⭐ Part 4 — The Most Important Orientation Rule

Always ask:

```text
What does a row mean?

What does a column mean?
```

In the convention we will use:

```text
Row
→ one target timestep

Column
→ one source position
```

So:

```text
A[t, i]
=
α_(t,i)
```

means:

> how much target step `t` attends to source position `i`.

---

# ⚠️ Part 5 — Visualization Libraries May Transpose It

Some diagrams or libraries may put:

```text
source positions on rows
target positions on columns
```

instead.

That is fine.

The mathematics has not changed.

So never memorize:

> source is always on x-axis.

Instead check the labels.

A strong interview answer is:

> The orientation of the plotted heatmap is a visualization convention; the important quantity is `α_(t,i)`, connecting target timestep `t` with source position `i`.

---

# 🧮 Part 6 — Small Alignment Matrix Example

Suppose:

```text
Source:
I    love    deep    learning
```

and target:

```text
Je   aime    apprentissage
```

For simplicity, imagine the model learns:

```text
          I     love   deep   learning

Je       0.80   0.10   0.05    0.05

aime     0.05   0.80   0.10    0.05

appr.    0.02   0.03   0.30    0.65
```

Each row approximately sums to:

```text
1
```

because each row is a softmax distribution over source positions.

---

# 🧠 Part 7 — How to Read the First Row

First target word:

```text
Je
```

has:

```text
[0.80, 0.10, 0.05, 0.05]
```

So it mostly retrieves information from:

```text
I
```

We could describe the soft alignment as:

```text
Je
↔
I
```

But remember:

> this is soft rather than strictly one-to-one.

---

# 🧠 Part 8 — How to Read the Third Row

Suppose:

```text
apprentissage
```

has:

```text
[0.02, 0.03, 0.30, 0.65]
```

This target representation receives substantial contribution from:

```text
deep
+
learning
```

So attention can naturally represent:

> phrase-level correspondence.

It does not have to align one target token to exactly one source token.

---

# ⭐ Part 9 — Alignment Is Soft

Traditional hard alignment might say:

```text
target_1 → source_1
target_2 → source_2
target_3 → source_4
```

Attention instead may say:

```text
target_3
→ 30% source_3
→ 65% source_4
→ 5% elsewhere
```

Conceptually:

```text
Hard Alignment
→ choose one position

Soft Attention Alignment
→ distribute relevance across positions
```

---

# 📐 Part 10 — Why Does Each Row Sum to 1?

For each fixed decoder timestep:

```text
t
```

we compute:

```text
α_(t,i)
=
exp(e_(t,i))
/
Σ_j exp(e_(t,j))
```

Therefore:

```text
Σ_i α_(t,i) = 1
```

So every row of the standard attention matrix represents:

> one probability-like distribution across source positions.

---

# ⚠️ Part 11 — Columns Do NOT Necessarily Sum to 1

This is a common mistake.

Because softmax is applied:

```text
across source positions
for each target timestep
```

we know:

```text
row sum = 1
```

But generally:

```text
column sum ≠ 1
```

A source position may receive:

* lots of attention across many target steps;
* little attention;
* repeated attention.

There is no standard column-normalization requirement.

---

# 📖 Part 12 — Turning the Matrix into a Heatmap

Numbers are useful, but large matrices quickly become difficult to read.

Instead we represent the weights visually.

For example:

```text
0.01 → very light
0.20 → light
0.60 → darker
0.95 → very dark
```

Conceptually:

```text
Attention Matrix
↓
Map weight magnitude to visual intensity
↓
Attention Heatmap
```

Higher intensity means:

> larger attention weight.

---

# 🧠 Part 13 — What Does a Heatmap Tell Us?

A heatmap lets us quickly inspect:

```text
Which source positions receive attention?

At which target timestep?

Does attention move through the source?

Does it jump backward?

Does it spread across phrases?

Does it repeatedly focus on one place?
```

These patterns can be very informative.

---

# 📖 Part 14 — Diagonal Alignment

Consider translation between languages with similar word order.

Source:

```text
I       love       AI
```

Target:

```text
Je      aime       IA
```

An attention matrix may look approximately:

```text
        I      love      AI

Je     0.85    0.10     0.05

aime   0.05    0.90     0.05

IA     0.05    0.10     0.85
```

Visually, high values form something close to:

> a diagonal line.

---

# ⭐ Part 15 — What Does a Diagonal Mean?

A diagonal pattern often suggests:

```text
source position 1
→ target position 1

source position 2
→ target position 2

source position 3
→ target position 3
```

So:

> source and target ordering are approximately aligned.

But language does not always preserve word order.

That is where attention becomes even more interesting.

---

# 📖 Part 16 — Reordering

Suppose one language orders information as:

```text
A B C
```

while the target requires:

```text
A C B
```

Attention may produce:

```text
        A      B      C

A'     0.9    0.05   0.05

C'     0.05   0.10   0.85

B'     0.05   0.85   0.10
```

Now the strong weights no longer form a simple diagonal.

Instead they:

> jump between source positions.

---

# 🧠 Part 17 — Why Reordering Matters

The decoder does not have to consume source information strictly:

```text
left → right
```

Attention gives it dynamic source access.

So a target timestep can retrieve:

```text
earlier source position

later source position

then earlier again
```

depending on what the target language requires.

This was one major advantage of attention for machine translation.

---

# ⭐ Part 18 — Attention Is Not Monotonic by Default

Standard global attention does not enforce:

```text
i_(t+1) ≥ i_t
```

The alignment can move:

* forward;
* backward;
* stay near the same region;
* revisit previous locations.

This flexibility helps model linguistic reordering.

---

# 📖 Part 19 — Sharp Attention

Suppose:

```text
α_t
=
[0.01, 0.02, 0.94, 0.03]
```

This is:

> **sharp / concentrated attention**

Most relevance is concentrated on:

```text
one source position
```

The corresponding heatmap row would have one very strong cell.

---

# 🧠 Part 20 — What Might Sharp Attention Mean?

Sharp attention may indicate that:

> one source representation is especially useful for the current prediction.

For example:

```text
person name
date
number
specific content word
```

may sometimes produce highly concentrated alignment.

But:

> sharp does not automatically mean correct.

---

# 📖 Part 21 — Diffuse Attention

Suppose:

```text
α_t
=
[0.20, 0.25, 0.30, 0.25]
```

This is:

> **diffuse attention**

The model is spreading source relevance across several positions.

---

# 🧠 Part 22 — Is Diffuse Attention Bad?

Not necessarily.

Some target predictions may depend on:

```text
multiple words
a phrase
overall sentence structure
longer contextual evidence
```

So diffuse attention can be completely reasonable.

Do not make the mistake:

```text
sharp attention = good

diffuse attention = bad
```

The correct interpretation depends on:

> task + timestep + model behavior.

---

# 📖 Part 23 — Phrase-Level Alignment

Suppose source:

```text
New   York   City
```

and the target language expresses the phrase with:

```text
one or two different tokens
```

The target token may attend to:

```text
New
+
York
+
City
```

simultaneously.

Therefore:

> alignment need not be word-to-word.

It can be:

```text
many-to-one
one-to-many
many-to-many
```

at the level of soft weights.

---

# 📖 Part 24 — Function Words

Consider words like:

```text
the
a
of
is
```

Their translations may depend strongly on:

* grammatical context;
* nearby words;
* target-language rules.

The attention distribution may therefore be:

> more distributed than for a concrete entity.

This is another reason not to expect every row to contain a perfect single peak.

---

# 📖 Part 25 — Repeated Attention

Suppose several consecutive target steps all strongly attend to:

```text
source position 4
```

For example:

```text
t1 → source 4 = 0.75
t2 → source 4 = 0.80
t3 → source 4 = 0.72
```

This may mean:

> the same source representation supports multiple target tokens.

That can be valid.

---

# ⚠️ Part 26 — Repeated Attention Can Also Reveal Problems

In some generation tasks, excessive repeated focus may correspond to:

* repeated phrases;
* stuck decoding;
* duplication;
* poor coverage of the source.

So visualization can help identify:

```text
decoder repeatedly attends to one location
↓
output repeatedly generates similar content
```

But this is a diagnostic clue, not proof by itself.

---

# 📖 Part 27 — Ignored Source Positions

Imagine a long source where one important date receives:

```text
almost zero attention
```

throughout decoding.

And the generated target also omits the date.

That is useful evidence.

We might investigate:

> Did the model fail to access the source representation?

But again, we must distinguish:

```text
attention evidence
```

from:

```text
causal proof
```

---

# ⭐ Part 28 — Alignment Visualization as a Debugging Tool

Attention heatmaps can help detect:

* important source positions receiving little attention;
* padding receiving attention;
* excessive repeated focus;
* unexpected reordering;
* overly diffuse attention;
* alignment failures on long inputs;
* source positions never used.

This makes visualization useful during:

> model development and error analysis.

---

# 🛡️ Part 29 — Padding in the Heatmap

Suppose the source is:

```text
I love AI PAD PAD
```

Correct masking should cause:

```text
α_PAD ≈ 0
```

So the final two columns should remain:

> nearly empty / very low intensity

across all decoder steps.

If you see strong PAD attention:

> that is a major implementation warning.

---

# 🧠 Part 30 — Heatmap Debugging Example

Suppose:

```text
Source:
The contract ends Friday PAD PAD
```

and attention repeatedly shows:

```text
PAD column 1 → 0.15
PAD column 2 → 0.20
```

That suggests:

> masking may be incorrect or missing.

A heatmap can expose this visually much faster than inspecting individual tensors.

---

# 📖 Part 31 — Alignment with `<EOS>`

The final target token:

```text
<EOS>
```

may have a less obvious alignment than a lexical word.

Why?

Because stopping generation may depend on:

* overall decoder state;
* source coverage;
* sequence semantics;
* previously generated target history.

So do not expect:

```text
<EOS> ↔ exactly one source word
```

---

# 📖 Part 32 — Alignment with `<BOS>` / `<SOS>`

Typically `<BOS>` is:

> an input to start decoding

rather than a lexical source-target alignment target in the same sense as content words.

When plotting attention maps, implementations may differ in whether special tokens are included.

Always inspect:

* source labels;
* target labels;
* whether BOS/EOS/PAD are plotted.

---

# 🧠 Part 33 — Attention Alignment Is Learned Indirectly

Recall the training chain:

```text
Attention Scores
↓
Softmax
↓
Context
↓
Decoder Prediction
↓
Loss
```

There may be no explicit label saying:

```text
target word 5 should attend to source word 8
```

Instead:

```text
better source access
↓
better prediction
↓
lower loss
```

causes useful alignment behavior to emerge.

---

# ⭐ Part 34 — Alignment Is an Emergent Internal Behavior

This is an important idea.

The model is trained for:

> output prediction.

Yet during that process it may learn something resembling:

> source-target alignment.

So alignment is often:

> an emergent intermediate behavior

rather than the primary supervised objective.

---

# 📐 Part 35 — Attention Matrix Dimensions Example

Suppose:

```text
source length = 8

target length = 5
```

Then:

```text
A ∈ R^(5 × 8)
```

using our convention.

That means:

```text
5 rows
→ target steps

8 columns
→ source positions
```

Each row contains 8 attention weights.

---

# 🧮 Part 36 — How Many Attention Values?

For:

```text
T_src = 8
T_tgt = 5
```

the alignment matrix contains:

```text
8 × 5 = 40
```

attention values.

In general:

```text
T_src × T_tgt
```

weights are generated across the full decoding process for global attention.

---

# 🧠 Part 37 — Connection to Computational Cost

This is the same structural reason classical global attention has approximately:

```text
O(T_src × T_tgt)
```

source-target score interactions.

The matrix visualization is not merely a picture.

It reflects:

> the pairwise alignment computations occurring during decoding.

---

# 📖 Part 38 — Source Coverage

Suppose we sum attention received by source position `i` across target timesteps:

```text
coverage_i
=
Σ_t α_(t,i)
```

This gives a rough idea of:

> how much total attention a source position received during decoding.

Notice:

```text
coverage_i
```

is not constrained to 1.

---

# 🧠 Part 39 — Coverage Intuition

If:

```text
coverage_i ≈ 0
```

then the source position was rarely attended to.

If:

```text
coverage_i
```

is large, it may have been attended to repeatedly.

This concept motivated techniques in some sequence-generation systems aimed at reducing:

* omission;
* repetition.

---

# ⚠️ Part 40 — Coverage Is Not the Same as Correct Translation

A source word receiving attention does not guarantee:

> it was translated correctly.

Likewise, low direct attention does not always prove:

> the information had no effect.

Why?

Because encoder states themselves may contain:

> contextual information from neighboring positions.

Especially with:

```text
BiRNN / BiLSTM
```

each `h_i` may represent more than one isolated word.

---

# ⭐ Part 41 — Encoder State ≠ Raw Source Token

This is crucial for interpreting heatmaps.

When we say:

```text
attention on source token i
```

more precisely we mean:

> attention on encoder representation `h_i`.

And:

```text
h_i
```

may encode:

* token `i`;
* surrounding context;
* sentence information.

So the heatmap is not a perfect map of:

> raw word importance.

---

# 📖 Part 42 — Why Bidirectional Encoders Complicate Interpretation

With a BiLSTM:

```text
h_i
=
[→h_i ; ←h_i]
```

The representation at position `i` contains information from:

```text
left context
+
right context
```

Therefore:

```text
high α_(t,i)
```

means the model strongly uses:

> the contextual representation centered at source position `i`.

It does not necessarily mean:

> only the visible token at position `i` matters.

---

# 🚨 Part 43 — The Biggest Interpretation Mistake

Suppose the model predicts:

```text
Friday
```

and gives attention:

```text
Friday source position = 0.9
```

It is tempting to conclude:

> The model predicted Friday because that source word causally caused the output.

Attention alone does not establish that.

A safer statement is:

> The corresponding encoder representation received a large weight in the context computation for that decoder step.

That is supported directly by the mechanism.

---

# 🧠 Part 44 — Contribution vs Causation

Attention weight tells us something about:

> contribution to the attention-weighted context.

It does not completely describe:

> causal importance to the final prediction.

Why?

Final output also depends on:

```text
decoder state
+
previous target history
+
encoder representations
+
attention context
+
output layers
+
learned nonlinear interactions
```

So:

```text
high attention
≠
complete causal explanation
```

---

# 📖 Part 45 — Counterfactual Thinking

A stronger causal question would be:

> What happens to the output if this source information or representation is removed or changed?

That requires interventions such as:

* perturbation;
* ablation;
* counterfactual analysis.

Simply observing a large attention value does not answer that question completely.

---

# ⭐ Part 46 — Correct Interview Language

Avoid:

> “Attention tells us exactly why the model made the prediction.”

Better:

> “Attention weights show how the model weighted encoder representations when constructing the context vector, so they can provide useful alignment and diagnostic signals, but they should not automatically be interpreted as complete causal explanations.”

That is a much stronger answer.

---

# 📖 Part 47 — Heatmaps Can Still Be Extremely Useful

The warning about causality does **not** mean visualization is useless.

Attention maps can still answer important questions:

```text
Where is the model focusing?

Does alignment look plausible?

Is padding masked?

Does the model revisit positions?

Does it ignore key regions?

Does word reordering appear?
```

These are valuable engineering signals.

---

# 🧠 Part 48 — Comparing Two Models

Suppose:

```text
Model A
→ fixed context Seq2Seq

Model B
→ attention-based Seq2Seq
```

With Model A there is no dynamic source-position distribution to inspect.

Model B gives:

```text
α_(t,i)
```

at each decoder timestep.

This adds a useful window into:

> source-decoder communication.

That was an important practical benefit of classical attention systems.

---

# 📖 Part 49 — Comparing Bahdanau and Luong Heatmaps

Both mechanisms ultimately generate:

```text
α_(t,i)
```

So both can produce alignment heatmaps.

The difference lies mainly in how the raw score was calculated:

```text
Bahdanau
→ additive scorer

Luong
→ multiplicative scorer
```

Once softmax produces `α`, visualization works the same way.

---

# ⭐ Part 50 — Visualization Does Not Tell You Which Scorer Was Used

If someone gives you only the final attention matrix:

```text
A
```

you usually cannot infer simply from that matrix whether it came from:

* Bahdanau;
* Luong Dot;
* Luong General.

Why?

Because the matrix contains:

> normalized attention outputs

not the internal scoring-function architecture.

---

# 📖 Part 51 — Alignment Evolution During Training

At the beginning of training, attention maps may look:

> noisy or nearly random.

As training improves, structured patterns may emerge:

```text
early:
diffuse / noisy

later:
task-relevant alignment patterns
```

This can make visualization useful for understanding:

> training progress.

But again, visual neatness is not itself the training objective.

---

# ⚠️ Part 52 — Pretty Heatmap ≠ Better Model

A model with a beautiful diagonal attention map is not automatically better than a model with a less visually clean map.

Evaluation must ultimately consider:

* task quality;
* accuracy;
* translation quality;
* factual preservation;
* relevant production metrics.

The heatmap is:

> diagnostic evidence

not:

> the final quality metric.

---

# 📖 Part 53 — Sharp vs Diffuse Revisited

Consider two distributions.

### Model A

```text
[0.01, 0.01, 0.96, 0.02]
```

### Model B

```text
[0.10, 0.25, 0.40, 0.25]
```

Without context, you cannot simply say:

```text
Model A is better
```

because the correct target might require:

* one source state;
* several source states.

Interpretation must be task-aware.

---

# 📐 Part 54 — Attention Entropy Intuition

One way to describe concentration mathematically is with entropy.

For a distribution:

```text
α_t
```

entropy is:

```text
H(α_t)
=
- Σ_i α_(t,i) log α_(t,i)
```

Conceptually:

```text
low entropy
→ concentrated / sharp attention

high entropy
→ distributed / diffuse attention
```

You do not need this metric to understand attention, but it gives us a mathematical way to discuss concentration.

---

# 🧠 Part 55 — Example: Low Entropy

```text
α =
[0.98, 0.01, 0.01]
```

Most probability mass sits in one place.

Therefore:

```text
entropy
→ relatively low
```

---

# 🧠 Part 56 — Example: High Entropy

```text
α =
[0.34, 0.33, 0.33]
```

The distribution is nearly uniform.

Therefore:

```text
entropy
→ relatively high
```

Again:

> neither is inherently correct or incorrect without task context.

---

# 📖 Part 57 — Uniform Attention

Suppose every target timestep produces:

```text
[0.25, 0.25, 0.25, 0.25]
```

This means:

> the model is not strongly differentiating source positions through attention.

That might be appropriate in rare situations, but persistent uniform attention could also suggest:

* weak alignment learning;
* scoring-network issues;
* early training;
* task not requiring localized access.

It deserves investigation.

---

# 📖 Part 58 — Attention Collapse

Another possible pattern is:

```text
every target step
→ almost always attends to the same source position
```

For example:

```text
t1 → [0.02, 0.95, 0.02, 0.01]
t2 → [0.01, 0.96, 0.02, 0.01]
t3 → [0.02, 0.94, 0.03, 0.01]
```

This might indicate:

> overly concentrated or collapsed attention behavior.

Whether it is truly problematic depends on the task and output.

---

# 🧠 Part 59 — Visualization-Based Debugging Workflow

A useful debugging process is:

```text
Bad Output
↓
Inspect Attention Map
↓
Identify suspicious alignment pattern
↓
Form hypothesis
↓
Inspect model / masks / data / representations
↓
Test hypothesis
```

The important point:

> attention visualization helps generate hypotheses.

It should not end the investigation by itself.

---

# 📖 Part 60 — Example Debugging Case

Source:

```text
The contract value is 2.4 million dollars
```

Generated output omits:

```text
2.4 million
```

Attention map shows almost no attention to the corresponding source region.

Possible hypothesis:

> decoder did not sufficiently retrieve the numeric representation.

But we should also inspect:

* encoder representation quality;
* training examples;
* decoding behavior;
* output probabilities;
* masking.

---

# ⭐ Part 61 — Another Debugging Case

Suppose generated text repeatedly outputs:

```text
the company the company the company
```

and the attention map repeatedly focuses on:

```text
same source region
```

That repeated alignment may support a hypothesis of:

> insufficient coverage / repeated source access.

Again:

> useful evidence, not complete proof.

---

# 📖 Part 62 — Alignment and Long Sequences

Recall the original fixed-context problem:

```text
long source
↓
one compressed context
↓
information loss / poor access
```

Attention allows direct access to source states.

Heatmaps make that access visible.

We can inspect whether later target steps still retrieve:

```text
early source positions
```

which was difficult with one fixed context vector.

---

# 🧠 Part 63 — Why This Matters Historically

Attention did more than improve translation accuracy.

It also gave researchers a way to observe:

> learned soft source-target alignments.

This provided intuitive evidence that the decoder was dynamically retrieving source information rather than relying only on one global summary.

---

# 📖 Part 64 — But Attention Is Still External to the Recurrent Core

At this point our architecture is still approximately:

```text
Encoder RNN
↓
h_1 ... h_T
       ↑
    Attention
       ↑
Decoder RNN
```

Attention determines:

> which encoder states to retrieve.

But:

```text
encoder recurrence
+
decoder recurrence
```

still exist.

This will soon become important.

---

# 🔗 Part 65 — Connection to Luong Global Attention

With global Luong attention:

```text
each decoder step
↓
scores every source position
```

So the alignment matrix is dense in principle:

```text
T_tgt × T_src
```

even if many learned weights become close to zero.

This gives full-source dynamic accessibility.

---

# 🔗 Part 66 — Connection to Local Attention

With local attention, only a source window may be considered.

Then the heatmap may show:

> narrow bands around selected source regions.

This reduces computation but constrains possible alignments.

So visualization can reveal the consequence of:

```text
global vs local source access
```

---

# 📖 Part 67 — Hard Alignment vs Soft Alignment

Let's summarize.

### Hard

```text
target timestep
↓
select exactly one source position
```

Example:

```text
[0, 0, 1, 0]
```

### Soft

```text
target timestep
↓
weighted contribution from source positions
```

Example:

```text
[0.05, 0.15, 0.65, 0.15]
```

Classical neural attention generally uses:

> soft attention

because it is differentiable.

---

# 🧠 Part 68 — Why Differentiability Matters Again

A hard `argmax` choice:

```text
choose source position 3
```

is not smoothly differentiable in the same way.

Soft attention:

```text
weighted sum
```

allows gradients to flow through:

```text
context
↓
weights
↓
scores
↓
attention parameters
```

So soft alignment can be trained end-to-end with backpropagation.

---

# ⭐ Part 69 — Alignment Is Not a Separate Module

Do not think the architecture contains:

```text
Attention
+
Alignment Module
```

The alignment is:

> the attention distribution itself viewed across source-target positions.

In other words:

```text
attention weights over all target steps
=
soft alignment matrix
```

---

# 📐 Part 70 — Compact Mathematical Summary

For each target timestep:

```text
e_(t,i)
=
score(s_t, h_i)
```

or the appropriate decoder state convention.

Normalize:

```text
α_(t,i)
=
exp(e_(t,i))
/
Σ_j exp(e_(t,j))
```

Context:

```text
c_t
=
Σ_i α_(t,i) h_i
```

Stack the weights:

```text
A_(t,i)
=
α_(t,i)
```

Then:

```text
A ∈ R^(T_tgt × T_src)
```

That matrix is what we visualize.

---

# 🎤 30-Second Interview Answer

> **Attention alignment is obtained by collecting the attention distribution from every decoder timestep. If the target length is `T_tgt` and the source length is `T_src`, the weights form a `T_tgt × T_src` matrix, where each row shows how one target timestep distributes attention across source positions. Visualizing this matrix as a heatmap can reveal patterns such as monotonic alignment, reordering, sharp or diffuse attention, ignored source regions, and masking problems. However, attention weights represent contribution to the attention context and should not automatically be interpreted as complete causal explanations of the final prediction.**

---

# 🎤 What Does One Cell Mean?

> `α_(t,i)` is the normalized attention weight assigned to encoder representation `h_i` while constructing the context for target timestep `t`.

---

# 🎤 Do Rows and Columns Both Sum to 1?

> **No.** With the standard source-wise softmax, each target-step row sums to 1. Columns generally do not, because the same source position may receive varying amounts of attention across multiple target timesteps.

---

# 🎤 Can Attention Weights Explain the Model?

A strong answer:

> **They provide useful alignment and diagnostic information because they directly determine how encoder states are weighted in the context vector. But the final prediction also depends on the decoder state and later model computations, so attention weights alone are not a complete causal explanation.**

---

# 🚫 High-Yield Traps

### ❌ Every target word aligns to exactly one source word

No.

Attention is generally soft.

---

### ❌ Every column of the attention matrix sums to 1

No.

Rows do under the standard source-wise softmax.

---

### ❌ Sharp attention always means good attention

No.

---

### ❌ Diffuse attention always means bad attention

No.

---

### ❌ Heatmap orientation is universally fixed

No.

Check axis labels.

---

### ❌ High attention weight proves causal importance

No.

---

### ❌ `h_i` represents only the raw source token

Not necessarily.

Especially with contextual/bidirectional encoders.

---

### ❌ Attention visualization is useless because it is not perfect explainability

No.

It remains very useful for alignment analysis and debugging.

---

# 📐 Formula Card

### Attention Weight

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

### Row Normalization

```text
Σ_i A_(t,i)
=
1
```

### Coverage Intuition

```text
coverage_i
=
Σ_t α_(t,i)
```

### Attention Entropy

```text
H(α_t)
=
-Σ_i α_(t,i) log α_(t,i)
```

---

# ⚡ 10 Things to Know Cold

1. One decoder timestep produces one attention distribution.
2. Stacking all distributions creates the alignment matrix.
3. Matrix shape is typically `T_tgt × T_src`.
4. Each row corresponds to one target timestep.
5. Each row normally sums to 1.
6. Diagonal patterns can indicate similar source-target ordering.
7. Off-diagonal patterns can represent reordering.
8. Attention can be sharp or diffuse; neither is universally better.
9. Heatmaps are useful for debugging and alignment analysis.
10. Attention weights are useful signals, not guaranteed causal explanations.

---

# 🧠 Final Mental Model

Start with one target step:

```text
Target Step t
↓
Attention over source
↓
α_t
```

Repeat for every target step:

```text
α_1
α_2
α_3
...
α_T
```

Stack them:

```text
        Source Positions
      x1 x2 x3 ... xN

y1    α  α  α     α
y2    α  α  α     α
y3    α  α  α     α
...
```

Then visualize:

```text
Attention Weights
↓
Alignment Matrix
↓
Heatmap
↓
Behavioral / Debugging Insight
```

But always remember:

# **Alignment signal ≠ complete causal explanation**

---

# 🔗 Next Lecture

Attention now gives us:

```text
dynamic source access
+
learned alignment
+
useful visualization
```

But our architecture still relies on:

```text
recurrent encoder
+
recurrent decoder
+
sequential processing
```

And global attention still requires many source-target comparisons.

So the next question is:

> **What limitations remain even after attention solves the fixed-context bottleneck?**

That leads to:

# **06_Attention_Limitations**

And those limitations will eventually create the need for:

# **Transformers**
