# 💡 Assignment — Bidirectional RNN

> **Goal:** Apply Bidirectional RNN concepts through sequence tracing, dimensional reasoning, leakage detection, architecture selection, and production scenarios.

---

# 🧩 Task 1 — Identify the Two Directions

Consider the sequence:

```text
x1, x2, x3, x4
```

### Questions

1. In what order does the forward RNN process the sequence?
2. In what order does the backward RNN process the sequence?
3. At position `x2`, what does each direction know?

### ✅ Answer

Forward:

```text
x1 → x2 → x3 → x4
```

Backward:

```text
x4 → x3 → x2 → x1
```

At `x2`:

```text
→h_2
→ knows x1 and x2
```

```text
←h_2
→ knows x4, x3, and x2
```

So the combined representation has context from:

> both sides of `x2`.

---

# 🧠 Task 2 — Ambiguous Word Reasoning

Consider:

```text
The fisherman sat on the bank near the river.
```

Suppose we want to classify the meaning of:

`bank`

### Question

Why can the backward RNN help?

### ✅ Answer

The forward RNN sees context such as:

```text
The fisherman sat on the bank
```

The backward RNN can use later words such as:

```text
near the river
```

That future context makes the river-bank meaning more obvious.

So:

```text
Forward state
→ left context

Backward state
→ right context
```

Together:

> they create a better contextual representation.

---

# 📐 Task 3 — Forward vs Backward Recurrence

Complete the equations.

### Forward

`→h_t = f(x_t, ________)`

### Backward

`←h_t = f(x_t, ________)`

### ✅ Answer

Forward:

`→h_t = f(x_t, →h_(t-1))`

Backward:

`←h_t = f(x_t, ←h_(t+1))`

Because:

```text
Forward
→ moves 1 → 2 → 3 → ...

Backward
→ moves T → T-1 → T-2 → ...
```

---

# 🧩 Task 4 — Two Networks or One?

A developer says:

> “A Bidirectional RNN uses the same RNN, first forward and then backward.”

### Correct or incorrect?

### ✅ Answer

Incorrect.

A Bidirectional RNN typically contains:

```text
Forward RNN
+
Backward RNN
```

with separate parameters.

So:

```text
Forward weights
≠
Backward weights
```

The two directions learn different sequence transformations.

---

# 📐 Task 5 — Hidden-State Dimension

Suppose:

```text
Forward hidden size = 64
Backward hidden size = 64
```

The two states are concatenated.

### Question

What is the output dimension at each position?

### ✅ Answer

`64 + 64 = 128`

So:

# `h_t ∈ R^128`

after concatenation.

---

# 🧮 Task 6 — Larger Hidden-State Example

Suppose each direction has:

`H = 256`

### Questions

1. What is the size of `→h_t`?
2. What is the size of `←h_t`?
3. What is the size after concatenation?

### ✅ Answer

```text
→h_t = 256
←h_t = 256
```

Combined:

# `512`

So a downstream Dense layer must expect:

> 512 input features per position

if concatenation is used.

---

# 🧠 Task 7 — Concatenation vs Addition

Suppose:

```text
→h_t = [0.3, 0.5]

←h_t = [-0.4, 0.8]
```

### Question 1

What is the concatenated representation?

### ✅ Answer

```text
[0.3, 0.5, -0.4, 0.8]
```

Dimension:

`4`

---

### Question 2

What would addition produce?

### ✅ Answer

```text
[0.3 + (-0.4),
 0.5 + 0.8]
```

Result:

```text
[-0.1, 1.3]
```

Dimension:

`2`

---

### Question 3

Why is concatenation often preferred?

### ✅ Answer

Because concatenation preserves:

> the forward and backward representations separately.

Addition immediately mixes them.

The downstream layer can learn later:

> how much each direction matters.

---

# 🔍 Task 8 — Which Context Does Each State Contain?

For:

```text
x1 x2 x3 x4 x5
```

at position:

`t = 3`

identify the context available to:

### `→h_3`

### `←h_3`

### ✅ Answer

Forward state:

```text
x1, x2, x3
```

Backward state:

```text
x5, x4, x3
```

So:

```text
[→h_3 ; ←h_3]
```

contains information related to the full sequence around position 3.

---

# 🧠 Task 9 — NER Scenario

Sentence:

```text
Amazon released a new cloud service.
```

Suppose the model needs to classify:

`Amazon`

as an organization.

### Question

How can bidirectionality help?

### ✅ Answer

The forward representation can use:

```text
Amazon
```

and previous context if present.

The backward representation can use later words:

```text
released a new cloud service
```

Those words strongly support:

> Amazon = organization/company

rather than some unrelated meaning.

So bidirectionality helps by using:

> right-side semantic evidence.

---

# 🚨 Task 10 — Detect Future Leakage

You are predicting whether a customer will churn on:

`1 January`.

The model input includes:

```text
customer activity from December
+
customer activity from January and February
```

A BiLSTM is used to classify churn as of:

`1 January`.

### Question

What is wrong?

### ✅ Answer

The model is using:

> events occurring after the prediction date.

January and February activity would not be available on 1 January.

That creates:

# **future leakage**

The model may perform extremely well offline because it effectively sees clues from the future.

At production inference time:

> those clues do not exist.

---

# ⭐ Task 11 — The Production-Time Question

Before using a Bidirectional RNN, what is the single most important question to ask?

### ✅ Answer

> **Will the future context used by the backward network actually be available when the real prediction is made?**

If yes:

> bidirectionality may be valid.

If no:

> use of future context creates leakage or requires deliberate prediction delay.

---

# 🌊 Task 12 — Streaming Sensor Data

A temperature sensor produces:

```text
x1, x2, x3, ...
```

You must detect an anomaly immediately when:

`x_t`

arrives.

### Question

Would a full Bidirectional RNN be appropriate?

### ✅ Answer

Usually no.

At time `t`, future observations:

```text
x_(t+1), x_(t+2), ...
```

do not exist yet.

So the backward representation cannot be computed using full future context.

A causal model such as:

* unidirectional RNN
* LSTM
* GRU

would be more appropriate.

---

# ⏳ Task 13 — Limited Lookahead

A speech-recognition system is allowed to delay transcription by:

`500 ms`.

### Question

Can it use some future context?

### ✅ Answer

Yes.

The system can buffer a short future window:

```text
Current frame
↓
wait 500 ms
↓
collect future frames
↓
predict
```

This creates a trade-off:

```text
More future context
→ potentially better accuracy

but

More future context
→ more latency
```

So bidirectional-like or limited-lookahead processing can be valid when:

> the latency budget allows it.

---

# 🧩 Task 14 — Is This Causal?

Classify each scenario as:

* ✅ bidirectional context may be valid
* ❌ full bidirectional context is inappropriate

### A. Named Entity Recognition on an uploaded document

### B. Next-token prediction in a chatbot

### C. Sentiment classification of a complete customer review

### D. Predicting stock price one minute into the future

### E. POS tagging on a completed sentence

### ✅ Answer

| Scenario                          | Result |
| --------------------------------- | ------ |
| A. NER on complete document       | ✅      |
| B. Next-token prediction          | ❌      |
| C. Complete review sentiment      | ✅      |
| D. Future stock forecasting       | ❌      |
| E. POS tagging completed sentence | ✅      |

The key distinction is:

> whether future sequence elements are legitimately available at inference time.

---

# 🔗 Task 15 — BiRNN vs BiLSTM

A developer says:

> “If I use a Bidirectional RNN, I no longer need LSTM because bidirectionality solves long-term dependencies.”

### Is this correct?

### ✅ Answer

No.

They address different problems.

```text
Bidirectionality
→ solves one-sided context limitation
```

```text
LSTM
→ improves long-term memory and gradient flow
```

Therefore:

```text
BiLSTM
=
Bidirectionality
+
LSTM memory mechanism
```

A Bidirectional vanilla RNN can still suffer from:

* vanishing gradients
* exploding gradients
* long-term dependency problems

---

# 🔗 Task 16 — BiGRU Reasoning

What does a BiGRU contain?

### ✅ Answer

```text
Forward GRU
+
Backward GRU
```

Each direction has its own:

* update gate
* reset gate
* hidden state
* parameters

At position `t`:

```text
[→h_t ; ←h_t]
```

is commonly used as the combined representation.

---

# 🧮 Task 17 — Parameter Count: Vanilla BiRNN

Suppose:

```text
Input dimension D = 32
Hidden dimension H = 64
```

One vanilla RNN direction has:

`H(D + H + 1)`

parameters.

Calculate the bidirectional parameter count.

### ✅ Answer

One direction:

`64 × (32 + 64 + 1)`

`= 64 × 97`

`= 6,208`

Bidirectional:

`2 × 6,208`

# `= 12,416`

excluding downstream layers.

---

# 🧮 Task 18 — Parameter Count: BiGRU

Using:

```text
D = 32
H = 64
```

Simplified one-direction GRU:

`3H(D+H+1)`

Calculate BiGRU parameters.

### ✅ Answer

One GRU direction:

`3 × 64 × 97`

`= 18,624`

Bidirectional:

`2 × 18,624`

# `= 37,248`

---

# 🧮 Task 19 — Parameter Count: BiLSTM

Using the same:

```text
D = 32
H = 64
```

Simplified one-direction LSTM:

`4H(D+H+1)`

### ✅ Answer

One direction:

`4 × 64 × 97`

`= 24,832`

Bidirectional:

`2 × 24,832`

# `= 49,664`

excluding downstream layers.

---

# 🧠 Task 20 — Why Do Parameters Roughly Double?

### ✅ Answer

Because bidirectionality creates:

```text
one forward recurrent model
+
one backward recurrent model
```

Each has:

> its own learned parameters.

So if one direction has:

`P`

parameters:

# `Bidirectional ≈ 2P`

---

# 🔄 Task 21 — Align the Outputs

Suppose:

```text
x1 x2 x3
```

Forward produces:

```text
→h1 →h2 →h3
```

Backward processing starts from:

`x3`

and produces corresponding states.

### Question

How should they be combined?

### ✅ Answer

They must be aligned by:

> original sequence position.

So:

```text
x1 → [→h1 ; ←h1]

x2 → [→h2 ; ←h2]

x3 → [→h3 ; ←h3]
```

Do not concatenate:

```text
→h1 with ←h3
```

just because those were computed first in their respective traversal directions.

---

# ⚠️ Task 22 — Debug Incorrect Alignment

A developer implements:

```text
Forward outputs:
[F1, F2, F3]

Backward outputs:
[B3, B2, B1]
```

and directly concatenates:

```text
[F1;B3]
[F2;B2]
[F3;B1]
```

### Is this correct?

### ✅ Answer

Not if:

`B3, B2, B1`

are still in reverse traversal order.

Backward outputs must first be aligned to the original positions:

```text
[B1, B2, B3]
```

Then:

```text
[F1;B1]
[F2;B2]
[F3;B3]
```

The key is:

> combine states representing the same original sequence position.

Frameworks often handle this alignment automatically, but conceptually it matters.

---

# 🧠 Task 23 — Whole-Sequence Representation

Suppose we want one representation for:

```text
x1 ... xT
```

Which two states can provide complete directional summaries?

### ✅ Answer

One common choice is:

```text
→h_T
+
←h_1
```

because:

`→h_T`

has traversed:

```text
x1 → ... → xT
```

and:

`←h_1`

has traversed:

```text
xT → ... → x1
```

So:

```text
[→h_T ; ←h_1]
```

can summarize the entire sequence from both directions.

---

# ⚠️ Task 24 — “Final Backward State” Trap

A developer says:

> “The full backward summary must be `←h_T` because T is the final token.”

### Correct?

### ✅ Answer

No.

The backward network begins at:

`x_T`

and ends at:

`x_1`.

So the backward state that has consumed the full reversed sequence corresponds to:

`←h_1`

in original indexing.

This is why:

> “final state” and “largest time index” are not always the same thing in a backward RNN.

---

# 🔄 Task 25 — Training

Does a Bidirectional RNN still use BPTT?

### ✅ Answer

Yes.

Both directions are recurrent.

So conceptually:

```text
Forward recurrence
→ BPTT

Backward recurrence
→ BPTT
```

The gradients update the parameters of both directional networks.

Bidirectionality does not replace:

> Backpropagation Through Time.

---

# ✅ True / False

| #  | Statement                                                          | Answer  |
| -- | ------------------------------------------------------------------ | ------- |
| 1  | A Bidirectional RNN normally contains two recurrent networks.      | ✅ True  |
| 2  | Forward and backward networks must share weights.                  | ❌ False |
| 3  | Forward hidden state uses past context.                            | ✅ True  |
| 4  | Backward hidden state can use later sequence positions.            | ✅ True  |
| 5  | Bidirectionality is itself a new recurrent cell type.              | ❌ False |
| 6  | BiLSTM is a Bidirectional RNN using LSTM cells.                    | ✅ True  |
| 7  | Bidirectional RNN automatically solves vanishing gradients.        | ❌ False |
| 8  | Concatenating two H-dimensional directional states gives `2H`.     | ✅ True  |
| 9  | Bidirectional RNN is always appropriate for streaming predictions. | ❌ False |
| 10 | Future leakage can make offline metrics misleading.                | ✅ True  |
| 11 | A backward RNN predicts unknown future inputs.                     | ❌ False |
| 12 | Full bidirectional processing generally requires future context.   | ✅ True  |
| 13 | BiRNN recurrent parameter count is roughly twice one direction.    | ✅ True  |
| 14 | Bidirectional RNN still uses BPTT.                                 | ✅ True  |
| 15 | Sequence order still matters in both directions.                   | ✅ True  |

---

# 🔥 Applied Leakage Challenge

Your training data contains customer events:

```text
Day 1
Day 2
Day 3
Day 4
Day 5
```

You want to predict on:

`Day 3`

whether the customer will make a purchase.

A BiLSTM uses all five days.

### Question 1

What information does the backward network provide at Day 3?

### ✅ Answer

It can use:

```text
Day 3
Day 4
Day 5
```

context.

---

### Question 2

If the prediction must happen during Day 3, what is the problem?

### ✅ Answer

Day 4 and Day 5 do not exist yet.

So the model uses:

> unavailable future information.

That is:

# future leakage

---

### Question 3

When could this same model be valid?

### ✅ Answer

If the task is an offline retrospective analysis where:

> all five days are already known.

For example:

> labeling historical behavior after the observation window has completed.

---

# ⭐ Staff Engineer Challenge — Offline Accuracy vs Production Reality

## Scenario

Your team builds two models for detecting support-ticket urgency.

### Model A

Unidirectional GRU

```text
F1 = 0.86
Latency = 30 ms
```

### Model B

BiGRU

```text
F1 = 0.92
Latency = 45 ms
```

The ticket text arrives gradually while the customer is typing.

The business wants urgency detection:

> as early as possible before the customer finishes typing.

---

## Question 1 — Should you automatically choose the BiGRU?

### ✅ Answer

No.

Although BiGRU has better offline F1, its backward direction requires:

> text that has not yet been typed.

So the important question is not only:

```text
Which model has higher F1?
```

It is:

```text
What information exists at prediction time?
```

---

## Question 2 — What architecture issue does this reveal?

### ✅ Answer

The production task is:

> causal / partial-sequence inference.

A fully bidirectional model assumes:

> full future text availability.

Therefore the offline evaluation may not match:

> real production conditions.

---

## Question 3 — What are possible alternatives?

### ✅ Answer

Depending on product requirements:

* use a unidirectional GRU/LSTM
* wait until the ticket is completed
* allow limited lookahead/buffering
* produce provisional predictions that update as more text arrives

The architecture should match:

> the actual inference-time contract.

---

## Question 4 — Suppose urgency is only classified after the complete ticket is submitted. Does that change the choice?

### ✅ Answer

Yes.

Now the full sequence is available.

So bidirectional processing becomes legitimate.

Then the decision can focus more directly on:

* accuracy
* latency
* memory
* compute
* operational cost

In that case, BiGRU's higher F1 may justify its additional cost.

---

## Question 5 — What general engineering principle does this demonstrate?

### ✅ Answer

> **Train-time information availability must match inference-time information availability.**

A model should not depend on information that production cannot provide at prediction time.

This principle extends far beyond Bidirectional RNNs.

---

# 🎯 Final Applied Exercise

Fill in the blanks:

```text
Normal RNN
↓
At position t sees __________ context
↓
May miss useful __________ context
↓
Bidirectional RNN
↓
Uses __________ RNN + __________ RNN
↓
Forward state = __________ context
Backward state = __________ context
↓
Combine them, often using __________
↓
Richer representation
```

### ✅ Answer

```text
Normal RNN
↓
At position t sees PAST context
↓
May miss useful FUTURE context
↓
Bidirectional RNN
↓
Uses FORWARD RNN + BACKWARD RNN
↓
Forward state = PAST context
Backward state = FUTURE context
↓
Combine them, often using CONCATENATION
↓
Richer representation
```

But:

```text
If future context is unavailable at inference
↓
Do NOT use full bidirectionality
```

---

# 🧠 Final Assignment Mental Model

```text
              x_t
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
Forward RNN        Backward RNN
       │               │
       ▼               ▼
     →h_t             ←h_t
       │               │
       └───────┬───────┘
               ▼
        [→h_t ; ←h_t]
               │
               ▼
     Context From Both Sides
```

Remember:

# **Forward = what came before**

# **Backward = what comes after**

# **Bidirectional = both sides**

And always ask:

# **“Will that future information really exist when I make the prediction?”**

---

# ⭐ Golden Rule

> **A Bidirectional RNN is useful only when right-side context is legitimately available: richer offline accuracy is meaningless if the backward network relies on future information that production cannot provide.**
