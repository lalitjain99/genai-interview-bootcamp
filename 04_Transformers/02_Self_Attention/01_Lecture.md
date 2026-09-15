# 📘 Self-Attention

## *How a Token Learns from Other Tokens in the Same Sequence*

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to answer:

* What is Self-Attention?
* Why was Self-Attention needed after classical Attention?
* How is Self-Attention different from encoder-decoder attention?
* What does it mean for one token to "attend" to another token?
* How does each token build a new contextual representation?
* Why are attention weights different for different tokens?
* Why can Self-Attention model long-range relationships?
* Why does Self-Attention enable better parallelism than recurrence?
* What is the output of a Self-Attention layer?
* Why do we need a weighted combination rather than selecting one token?
* Why is Self-Attention not enough by itself to understand word order?
* What question naturally leads us to Query, Key, and Value?

---

# 🌍 Chapter 1 — Where We Left Off

In the previous lecture, we ended with an important question.

We already knew that classical Attention allowed:

```text
Decoder State
        ↓
attends to
        ↓
Encoder States
```

This solved the fixed-context bottleneck.

But the encoder itself was still recurrent.

So source-side communication looked roughly like:

```text
word 1
  ↓
word 2
  ↓
word 3
  ↓
...
  ↓
word T
```

Then we asked:

> If Attention already allows one representation to directly retrieve information from another...

why should source tokens still communicate only through recurrence?

That question leads directly to:

# **Self-Attention**

---

# 🌍 Chapter 2 — Start with a Simple Sentence

Consider:

> **The animal didn't cross the street because it was tired.**

Focus on the token:

```text
it
```

To properly understand `it`, the model needs information from:

```text
animal
```

Now think like an engineer.

We want the representation of `it` to somehow become aware of `animal`.

How could that happen?

---

# 🌍 Chapter 3 — The Old RNN Way

In an RNN, information flows sequentially.

Conceptually:

```text
The
 ↓
animal
 ↓
didn't
 ↓
cross
 ↓
the
 ↓
street
 ↓
because
 ↓
it
```

Information about `animal` may gradually flow through several hidden states before reaching `it`.

That works.

But the path is long.

So in the previous lecture we asked:

> Can `it` directly inspect `animal`?

Self-Attention says:

# **Yes.**

---

# 🌍 Chapter 4 — The Core Idea

Instead of forcing information to travel step by step, let every token inspect the other tokens in the sequence.

For `it`:

```text
it
↓
look at "The"
look at "animal"
look at "didn't"
look at "cross"
look at "the"
look at "street"
look at "because"
look at "it"
look at "was"
look at "tired"
```

The token asks:

> Which other tokens contain information that is useful for understanding me?

Perhaps the model decides:

```text
animal   → very important
tired    → somewhat important
street   → less important
The      → very little importance
```

Then `it` gathers information from all of them.

This is the essence of Self-Attention.

---

# 💡 First Definition

Self-Attention is a mechanism where:

> **each token dynamically decides how much information to gather from other tokens in the same sequence.**

The important phrase is:

# **same sequence**

That is why it is called:

# **Self-Attention**

---

# 🌍 Chapter 5 — Classical Attention vs Self-Attention

Let's compare.

## Classical Encoder-Decoder Attention

```text
Decoder State
      ↓
attends to
      ↓
Encoder States
```

Two different sides are involved:

```text
Target
→
Source
```

---

## Self-Attention

```text
Token Representation
       ↓
attends to
       ↓
Representations from the Same Sequence
```

Conceptually:

```text
Source
→
Source
```

or later in a decoder:

```text
Target
→
Target
```

The important difference is not the word "attention."

The important difference is:

> **where the information comes from.**

---

# 🌍 Chapter 6 — What Does "Attend" Actually Mean?

The word "attend" sounds abstract.

Let's make it concrete.

Suppose our sentence contains four tokens:

```text
The
cat
sat
there
```

Take the token:

```text
sat
```

Self-Attention may assign relevance values like:

```text
The    → 0.05
cat    → 0.60
sat    → 0.20
there  → 0.15
```

These numbers tell us how strongly `sat` gathers information from each token.

After normalization:

```text
0.05 + 0.60 + 0.20 + 0.15 = 1
```

These are:

# **Attention Weights**

---

# 🧠 Chapter 7 — Why Does a Token Attend to Itself?

You may notice:

```text
sat → sat
```

is also included.

Why?

Because the token's own information is still useful.

Self-Attention does not mean:

> "ignore yourself and only look elsewhere."

It means:

> "build a better representation using yourself plus relevant information from the rest of the sequence."

So the token may attend to:

```text
itself
+
other tokens
```

---

# 🌍 Chapter 8 — From Weights to a New Representation

Now suppose each token has a vector representation.

For simplicity:

```text
The   = h₁
cat   = h₂
sat   = h₃
there = h₄
```

For the token `sat`, suppose the attention weights are:

```text
0.05
0.60
0.20
0.15
```

Then its new contextual representation can be built as:

```text
new_sat
=
0.05h₁
+
0.60h₂
+
0.20h₃
+
0.15h₄
```

This is exactly the same weighted-sum idea we already learned in classical Attention.

---

# 🔗 Connecting Back to Classical Attention

Recall:

```text
c_t = Σᵢ α_(t,i) hᵢ
```

The decoder created a context vector from encoder states.

Self-Attention uses the same high-level pattern:

```text
output for token i
=
weighted combination
of token representations
```

The difference is:

```text
Classical Attention
→ decoder gathers from encoder states

Self-Attention
→ token gathers from same-sequence token states
```

So Self-Attention is not a completely new idea.

It is a powerful generalization of something we already know.

---

# 🧠 Chapter 9 — The Three-Step Mental Model

At a high level, every token does three things.

```text
1. Compare
↓
Which tokens are relevant to me?

2. Normalize
↓
Convert relevance scores into weights

3. Aggregate
↓
Combine information using those weights
```

So our old Attention mental model still works:

# **Compare → Normalize → Aggregate**

That continuity is important.

---

# 🌍 Chapter 10 — Different Tokens Need Different Information

Consider:

> **The animal didn't cross the street because it was tired.**

The token:

```text
it
```

may care strongly about:

```text
animal
```

But the token:

```text
cross
```

may care more about:

```text
animal
street
```

And:

```text
tired
```

may care about:

```text
animal
it
```

So there is no single attention distribution for the whole sentence.

Each token gets:

> **its own attention distribution**

over the sequence.

---

# 🧮 Chapter 11 — A Small Attention Matrix

Suppose we have three tokens:

```text
I
love
AI
```

Imagine the attention weights are:

```text
          I     love    AI

I       0.50    0.30   0.20
love    0.20    0.40   0.40
AI      0.10    0.30   0.60
```

Each row corresponds to:

> one token asking where to gather information from.

For example:

```text
AI
→ 10% from I
→ 30% from love
→ 60% from AI
```

Each row sums to:

```text
1
```

just like the attention distributions we studied before.

---

# 🧠 Chapter 12 — What Does One Cell Mean?

Consider:

```text
A[AI, love] = 0.30
```

This means:

> while building the new representation of `AI`, the model assigns weight 0.30 to the representation associated with `love`.

Notice what it does **not** mean.

It does not necessarily mean:

> "`love` caused everything the model knows about AI."

It simply means:

> that representation contributed with that attention weight in this attention computation.

Same caution we learned in Attention Visualization still applies.

---

# 🌍 Chapter 13 — Before and After Self-Attention

Before Self-Attention, suppose the token `bank` starts with an embedding representing the word:

```text
bank
```

But `bank` is ambiguous.

Consider:

```text
I deposited money in the bank.
```

versus:

```text
We sat on the bank of the river.
```

Initially, the word token may start from a similar base representation.

But through Self-Attention:

### Sentence 1

`bank` may gather information from:

```text
deposited
money
```

### Sentence 2

`bank` may gather information from:

```text
river
sat
```

So after attention:

```text
bank in sentence 1
≠
bank in sentence 2
```

The representation becomes:

# **Contextual**

---

# 🧠 Chapter 14 — Static Representation vs Contextual Representation

Before contextual processing:

```text
bank
↓
word representation
```

After Self-Attention:

```text
bank
+
context from nearby/distant words
↓
context-aware representation
```

This is a major idea.

Self-Attention does not merely tell us:

> where to look.

It helps create:

> a new representation informed by the rest of the sequence.

---

# 🌍 Chapter 15 — Why Weighted Combination Instead of Selecting One Token?

Why not simply choose the most relevant token?

For example:

```text
it
→
animal
```

and ignore everything else?

Because meaning is often distributed.

For `it`, perhaps:

```text
animal
```

tells us what it refers to,

while:

```text
tired
```

provides semantic context,

and:

```text
because
```

provides structural information.

So instead of:

```text
pick one token
```

Self-Attention uses:

```text
soft weighted retrieval
```

where several tokens can contribute.

This preserves differentiability and lets the model combine information.

---

# 🔗 Connection to Soft Attention

This is exactly why classical Attention used:

```text
softmax
```

rather than hard selection.

We can write conceptually:

```text
relevance scores
↓
softmax
↓
attention weights
↓
weighted sum
```

That idea survives directly into Self-Attention.

---

# 🌍 Chapter 16 — Long-Range Relationships Become Easier

Consider:

```text
The book that I bought during my trip to London last year was excellent.
```

The word:

```text
was
```

needs to relate to:

```text
book
```

even though many words appear between them.

In an RNN, information travels across multiple recurrent transitions.

With Self-Attention:

```text
was
──────────────→
book
```

can become a direct interaction.

This gives Self-Attention a powerful property:

# **Distance does not require a longer recurrent path.**

---

# 🧠 Chapter 17 — Important Nuance

This does not mean:

> Self-Attention automatically understands every long-distance dependency perfectly.

It means:

> the architecture provides a short computational path between distant positions.

Whether the model learns the correct relationship still depends on:

* training,
* learned parameters,
* data,
* depth,
* optimization.

Architecture makes the relationship easier to access.

It does not guarantee that the model will learn it perfectly.

---

# 🌍 Chapter 18 — Why Self-Attention Helps Parallelism

Now consider an input sequence:

```text
x₁
x₂
x₃
x₄
```

An RNN requires:

```text
h₁
↓
h₂
↓
h₃
↓
h₄
```

But Self-Attention does not require:

```text
output₂
```

to wait for:

```text
output₁
```

in the same way.

Each token can compute relevance against the available sequence representations.

Conceptually:

```text
x₁ ─┐
x₂ ─┼─→ interactions for all positions
x₃ ─┼─→ computed together
x₄ ─┘
```

That is why Self-Attention enables much greater:

# **sequence-level parallelism**

during training.

---

# 🧠 Chapter 19 — The Important Trade-Off

We have removed the recurrent chain.

Great.

But now every token may compare with every other token.

For:

```text
T
```

tokens:

```text
T tokens
×
T tokens
```

gives roughly:

```text
T²
```

pairwise relationships.

So full Self-Attention introduces roughly:

```text
O(T²)
```

attention interactions.

Again, architecture is a trade-off.

---

# 📊 RNN vs Self-Attention

| Property                  | RNN                     | Self-Attention              |
| ------------------------- | ----------------------- | --------------------------- |
| Token interaction         | Through recurrent chain | Direct pairwise interaction |
| Sequence processing       | Sequential              | Highly parallelizable       |
| Distant dependency path   | Long                    | Short                       |
| Built-in order            | Yes                     | No                          |
| Pairwise interaction cost | Lower all-pairs cost    | Roughly `O(T²)`             |

---

# 🌍 Chapter 20 — Wait... Where Did Order Go?

We have gained:

```text
direct interaction
+
parallelism
```

But remember the sentence:

```text
Dog bites man
```

and:

```text
Man bites dog
```

Same words.

Different order.

Different meaning.

An RNN naturally sees order because it processes:

```text
position 1
then
position 2
then
position 3
```

Self-Attention itself does not inherently know:

```text
which token came first
```

unless positional information is introduced.

So Self-Attention creates another problem:

# **How do we represent token position?**

We will solve that later with Positional Encoding.

---

# 🧠 Chapter 21 — But We Still Have a Bigger Missing Piece

So far we have said:

```text
Each token compares itself
with every other token.
```

But that sentence hides the most important mechanism.

How exactly does the comparison happen?

Take:

> **The animal didn't cross the street because it was tired.**

When processing:

```text
it
```

how does the model know that:

```text
animal
```

should receive a stronger score than:

```text
street
```

We need some learned mechanism for compatibility.

At first, we might think:

```text
just compare token embeddings
```

But that raises several questions.

---

# 🌍 Chapter 22 — One Representation Has to Do Too Many Jobs

Suppose the representation of `it` is:

```text
h_it
```

and the representation of `animal` is:

```text
h_animal
```

Could we just compute:

```text
h_itᵀ h_animal
```

?

Maybe.

But think about what a token representation needs to do.

When `it` is looking at other words, it needs to express:

> **What information am I looking for?**

When `animal` is being inspected, it needs to express:

> **What information do I contain that others may find relevant?**

And when its information is actually retrieved, it needs to provide:

> **What information should I send?**

These are three related but different roles.

Using one unchanged representation for all three roles may be restrictive.

---

# 💡 The Next Big Question

What if each token could create different learned representations for different purposes?

One representation for:

```text
What am I looking for?
```

Another for:

```text
What do I contain?
```

And another for:

```text
What information should I contribute?
```

Those three roles will become:

```text
Query
Key
Value
```

But now they are not random terms.

We have discovered **why they need to exist**.

---

# 🧠 Engineer's Insight

Notice the progression.

We did not start Self-Attention with:

```text
Q = XW_Q
K = XW_K
V = XW_V
```

because those equations do not explain the idea.

We first asked:

```text
How can one token learn from another?
        ↓
Give every token access to the sequence
        ↓
How much should it use from each token?
        ↓
Attention weights
        ↓
How should information be combined?
        ↓
Weighted sum
        ↓
How do tokens become context-aware?
        ↓
Self-Attention
```

Now a new limitation appears:

```text
How exactly do we represent
"what I need"
vs
"what I offer"?
```

That limitation naturally leads to the next chapter.

---

# 🧠 First-Principles Chain

```text
Classical Attention
        │
        ▼
Decoder can retrieve encoder information directly
        │
        ▼
Can tokens in the same sequence
retrieve from each other?
        │
        ▼
Self-Attention
        │
        ▼
Each token compares with other tokens
        │
        ▼
Relevance Scores
        │
        ▼
Normalize Scores
        │
        ▼
Attention Weights
        │
        ▼
Weighted Combination
        │
        ▼
Contextual Token Representation
        │
        ▼
But how should tokens represent
what they seek and what they provide?
        │
        ▼
Query / Key / Value
```

---

# 📐 Minimal Mathematical View

For token `i`, suppose we compute a compatibility score with token `j`:

```text
e_(i,j)
=
score(h_i, h_j)
```

Normalize over all `j`:

```text
α_(i,j)
=
exp(e_(i,j))
/
Σ_k exp(e_(i,k))
```

Then build the new representation:

```text
z_i
=
Σ_j α_(i,j) h_j
```

Where:

```text
e_(i,j)
→ raw relevance between token i and token j

α_(i,j)
→ normalized attention weight

z_i
→ contextualized output for token i
```

This is intentionally generic.

We have **not** yet introduced Query, Key, or Value.

---

# 🧮 Worked Example

Suppose:

```text
h₁ = [1, 0]
h₂ = [0, 1]
h₃ = [1, 1]
```

For token 2, suppose Self-Attention gives:

```text
α_(2,1) = 0.2
α_(2,2) = 0.3
α_(2,3) = 0.5
```

Then:

```text
z₂
=
0.2h₁
+
0.3h₂
+
0.5h₃
```

Substitute:

```text
z₂
=
0.2[1,0]
+
0.3[0,1]
+
0.5[1,1]
```

So:

```text
z₂
=
[0.2,0]
+
[0,0.3]
+
[0.5,0.5]
```

Therefore:

```text
z₂ = [0.7, 0.8]
```

The new representation of token 2 is no longer based only on token 2.

It now contains information gathered from:

```text
token 1
+
token 2
+
token 3
```

That is Self-Attention in its simplest form.

---

# 🚨 High-Yield Misconceptions

### ❌ Self-Attention means each token picks one other token

No.

It usually performs soft weighted aggregation.

---

### ❌ Every token uses the same attention weights

No.

Each token can have its own attention distribution.

---

### ❌ A token cannot attend to itself

It can.

---

### ❌ Self-Attention and encoder-decoder attention are identical

The generic mechanism is related, but the source of the representations differs.

---

### ❌ Self-Attention automatically knows word order

No.

Positional information is needed.

---

### ❌ Self-Attention completely eliminates computation cost

No.

Full Self-Attention introduces roughly `O(T²)` pairwise relationships.

---

### ❌ Self-Attention automatically understands long-range dependencies

No.

It provides short direct paths, but the relationships still have to be learned.

---

# 🎤 30-Second Interview Answer

> **Self-Attention allows each token in a sequence to dynamically gather information from other tokens in the same sequence. For every token, the model computes relevance scores against other positions, normalizes those scores into attention weights, and forms a weighted combination to create a contextualized representation. Unlike RNNs, distant positions can interact directly and many token positions can be processed in parallel. However, Self-Attention by itself does not inherently encode token order and full attention introduces roughly quadratic pairwise interactions.**

---

# 🎤 How Is Self-Attention Different from Classical Attention?

> **Classical encoder-decoder attention typically lets a decoder state retrieve information from encoder states. Self-Attention applies the same general compare-normalize-aggregate idea within one sequence, so each token can retrieve information from other tokens in that same sequence.**

---

# 🎤 Why Does Self-Attention Help Long-Range Dependencies?

> **Because two distant positions can interact directly within an attention layer instead of requiring information to propagate through every intermediate recurrent state. This creates much shorter dependency paths.**

---

# 🎯 Key Takeaways

* Self-Attention grew naturally from classical Attention.
* It allows tokens to retrieve information from other tokens in the same sequence.
* Every token can have its own attention distribution.
* Tokens can also attend to themselves.
* The generic pipeline remains:
  **Compare → Normalize → Aggregate**.
* The result is a new contextualized representation for every token.
* Self-Attention enables direct long-range interactions.
* It allows far greater sequence-level parallelism than recurrence.
* Full Self-Attention creates roughly `O(T²)` pairwise interactions.
* Self-Attention alone does not encode sequence order.
* We still have not answered how relevance should actually be parameterized.
* That missing idea leads naturally to **Query, Key, and Value**.

---

# 🔜 Looking Ahead

We now know what we want:

```text
Token
↓
look at other tokens
↓
decide relevance
↓
gather useful information
↓
create contextual representation
```

But there is still a hidden problem.

Suppose `it` is examining `animal`.

The representation of `it` needs to express:

> **What am I looking for?**

The representation of `animal` needs to express:

> **What information do I have that might match that need?**

And if `animal` is selected, we also need:

> **What information should actually be passed forward?**

Those are not necessarily the same thing.

So the next question becomes:

> **Can each token learn separate representations for searching, matching, and providing information?**

That leads directly to:

# 🚀 **03 — Query, Key, Value**
