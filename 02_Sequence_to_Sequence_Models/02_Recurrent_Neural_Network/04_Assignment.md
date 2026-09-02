# 💡 Assignment — Recurrent Neural Network

> **Goal:** Apply RNN concepts to realistic modeling and debugging situations.
> This assignment focuses on **reasoning**, not repeating definitions from the interview file.

---

# 🧩 Task 1 — Trace the RNN State

An RNN processes:

```text
A → B → C → D
```

Assume:

`h0 = 0`

and conceptually:

`h_t = f(x_t, h_{t-1})`

### Questions

1. Which inputs can influence `h1`?
2. Which inputs can influence `h2`?
3. Which inputs can influence `h4`?
4. Does `h4` necessarily preserve all information from `A` perfectly?

### ✅ Answer

### `h1`

Can be influenced by:

`A`

because:

`h1 = f(A, h0)`

---

### `h2`

Can be influenced by:

```text
A
B
```

because `h2` receives `h1`, and `h1` was influenced by `A`.

---

### `h4`

Can potentially be influenced by:

```text
A → B → C → D
```

through the recurrent chain.

But:

> influence does not mean perfect memory.

`h4` is a compressed learned representation, so information from early steps may be weakened or lost.

---

# 🔧 Task 2 — Debug the Architecture

A developer builds the following model:

```text
x1 → RNN_A → h1
x2 → RNN_B → h2
x3 → RNN_C → h3
```

where each RNN has its own independent weights.

### Question

What important property of a standard RNN has been lost?

### ✅ Answer

The model has lost:

> **parameter sharing across time**

A standard RNN should use:

```text
x1 → SAME RNN → h1
x2 → SAME RNN → h2
x3 → SAME RNN → h3
```

Using independent parameters per position:

* ties the model to sequence positions
* increases parameter count with sequence length
* prevents one common recurrent update rule from being learned

---

# 🧮 Task 3 — Design the RNN Dimensions

You are building a text classifier.

Each token embedding has:

`256 dimensions`

You choose:

`hidden size = 512`

### Questions

Find the shapes of:

* `W_xh`
* `W_hh`
* `b_h`

Then calculate the recurrent parameter count.

### ✅ Answer

Given:

`D = 256`

`H = 512`

### Shapes

`W_xh = 512 × 256`

`W_hh = 512 × 512`

`b_h = 512`

### Parameters

Input-to-hidden:

`512 × 256 = 131,072`

Hidden-to-hidden:

`512 × 512 = 262,144`

Bias:

`512`

Total:

`131,072 + 262,144 + 512`

# `= 393,728`

---

# 🧠 Task 4 — Which State Should Be Used?

You are building a sentiment classifier for:

```text
The food was surprisingly good
```

The RNN generates:

```text
h1, h2, h3, h4, h5
```

You want one final sentiment prediction.

### Question

Which state would a simple vanilla RNN classifier commonly use?

### ✅ Answer

A simple many-to-one design commonly uses:

`h5`

the final hidden state.

Then:

```text
h5
↓
Linear Layer
↓
Positive / Negative
```

The assumption is that the final hidden state summarizes useful sequence information.

This is also where the fixed-state bottleneck can become a limitation for long sequences.

---

# 🏷️ Task 5 — Choose the Correct RNN Output Pattern

Match each task with the most natural classical RNN pattern.

### A. Sentiment classification

### B. Named Entity Recognition

### C. Predicting a label for every sensor reading

### D. Reading a transaction history and producing one fraud-risk score

### ✅ Answer

```text
A → Many-to-One
B → Many-to-Many
C → Many-to-Many
D → Many-to-One
```

---

# 🔄 Task 6 — Same Weights, Different States

Suppose an RNN uses exactly the same:

`W_xh`

and:

`W_hh`

at every time step.

A developer says:

> “Then `h1`, `h2`, and `h3` must also be identical.”

### Explain why this is wrong.

### ✅ Answer

The function is shared, but its inputs are not.

At each time step:

```text
h_t = f(x_t, h_{t-1})
```

Both:

* `x_t`
* `h_{t-1}`

can change.

Therefore:

```text
Same Function
+
Different Inputs
=
Different Outputs
```

So parameter sharing does not imply identical hidden states.

---

# 🧩 Task 7 — Remove the Recurrent Connection

Consider:

`h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

Now remove:

`W_hh h_{t-1}`

### Question

What behavior does the network lose?

### ✅ Answer

The model loses the direct mechanism for using:

> previous hidden context.

The update becomes approximately:

`h_t = tanh(W_xh x_t + b_h)`

Now each hidden state depends primarily on the current input.

So recurrence is effectively removed.

---

# ⏳ Task 8 — Diagnose a Long-Term Dependency Failure

An RNN processes a sequence of 2,000 events.

The first event contains information critical to predicting the final output.

During training, the model consistently ignores that early event.

### Question

Give two RNN-specific reasons this can happen.

### ✅ Answer

### 1. Information Retention Problem

The information must survive:

```text
h1 → h2 → ... → h2000
```

Repeated state transformations can weaken or overwrite earlier information.

### 2. Gradient Propagation Problem

During BPTT, the learning signal must travel backward through many recurrent steps.

Repeated multiplications may cause:

> vanishing gradients

so early time steps receive very little useful learning signal.

---

# ⚡ Task 9 — Parameter Count vs Sequence Length

Model A:

```text
D = 100
H = 200
Sequence Length = 20
```

Model B:

```text
D = 100
H = 200
Sequence Length = 500
```

### Questions

1. Which model has more recurrent trainable parameters?
2. Which requires more recurrent computation?

### ✅ Answer

### Trainable Parameters

Both have the same number.

Because:

`D` and `H`

are identical and recurrent weights are shared.

### Computation

Model B requires more computation because the RNN cell runs:

`500`

times instead of:

`20`.

So:

```text
Same Parameters
Different Compute
```

---

# 🌊 Task 10 — Streaming Design

A machine sends one vibration measurement every 100 ms.

You maintain hidden state:

`h_t`

for anomaly detection.

At time `t+1`, what minimum information does the vanilla RNN need to continue processing?

### ✅ Answer

It needs:

* new measurement `x_{t+1}`
* previous hidden state `h_t`

Then:

`h_{t+1} = f(x_{t+1}, h_t)`

The full sequence does not need to be replayed.

This is a major advantage of recurrent state for:

> streaming inference.

---

# 🧠 Task 11 — State Belongs to Whom?

You deploy one shared RNN model for three users:

```text
User A
User B
User C
```

Should they all share the same hidden state?

### ✅ Answer

No.

They can share the same:

> model parameters

but each independent sequence should maintain its own hidden state.

```text
Shared Model Weights
↓
User A → h_A
User B → h_B
User C → h_C
```

Important:

```text
Parameters
→ global/shared

Hidden State
→ sequence-specific
```

---

# ⚠️ Task 12 — Hidden-State Leakage Bug

A production service processes:

```text
User A sequence
```

then immediately processes:

```text
User B sequence
```

without resetting or replacing the hidden state.

### Question

What problem can occur?

### ✅ Answer

User B's sequence can begin with context derived from:

> User A.

This causes:

* incorrect predictions
* sequence contamination
* possible data leakage across independent sequences

The hidden state must be:

> reset or loaded separately for each independent sequence.

---

# 🧮 Task 13 — Output Layer Design

Your RNN has:

`hidden size = 128`

You need to classify each time step into:

`10 classes`.

### Questions

What should the output weight matrix shape be?

How many output-layer parameters are there including bias?

### ✅ Answer

We need:

`128 → 10`

So:

`W_hy.shape = 10 × 128`

Bias:

`b_y.shape = 10`

Parameter count:

`10×128 + 10`

`= 1,280 + 10`

# `= 1,290`

---

# 🔍 Task 14 — Find the Incorrect Claim

Which statement is wrong?

### A.

A longer sequence requires more recurrent computations.

### B.

A longer sequence automatically requires more recurrent weights.

### C.

Hidden state changes across time.

### D.

RNN weights are reused across time.

### ✅ Answer

❌ **B is wrong.**

Sequence length affects:

> number of recurrent applications

not:

> number of learned recurrent parameters.

---

# 🧠 Task 15 — Can Vanilla RNN Understand Future Context?

Suppose the token:

```text
bank
```

appears in:

```text
He walked toward the bank of the river
```

At the moment a standard forward RNN processes `"bank"`, it has not yet seen:

```text
of the river
```

### Question

Can the hidden state at `"bank"` use this future information?

### ✅ Answer

No.

A standard forward RNN processes:

```text
left → right
```

At time `t`, it has access to:

```text
x1 ... x_t
```

not future inputs.

Using both past and future context motivates:

> Bidirectional RNNs

which we will study later.

---

# ✅ True / False — Reasoning Traps

| #  | Statement                                                                               | Answer  |
| -- | --------------------------------------------------------------------------------------- | ------- |
| 1  | `h_t` changes over time even though RNN weights are shared.                             | ✅ True  |
| 2  | Every sequence in a batch should use the exact same hidden-state values.                | ❌ False |
| 3  | A longer sequence increases RNN computation but not recurrent parameter count.          | ✅ True  |
| 4  | `W_hh` is responsible for connecting previous hidden context to the current state.      | ✅ True  |
| 5  | Removing `W_hh h_{t-1}` largely removes recurrence from a vanilla RNN.                  | ✅ True  |
| 6  | A hidden size of 128 guarantees perfect memory of the previous 128 tokens.              | ❌ False |
| 7  | An RNN can perform streaming inference without replaying all earlier inputs.            | ✅ True  |
| 8  | Standard forward RNN state can naturally use future tokens.                             | ❌ False |
| 9  | Shared RNN weights mean all sequence positions perform the same learned type of update. | ✅ True  |
| 10 | Hidden state and model parameters are the same thing.                                   | ❌ False |
| 11 | Separate user sequences can share weights while keeping separate hidden states.         | ✅ True  |
| 12 | Vanilla RNNs may struggle when relevant information is thousands of steps away.         | ✅ True  |

---

# ⭐ Staff Engineer Challenge — Stateful Streaming Service

You need to deploy an RNN-based fraud model.

There are:

`5 million users`

Each user's transactions arrive independently.

The RNN has:

`hidden size = 256`

The service must make a prediction within:

`50 ms`

for every transaction.

---

## Question 1

What state must the production system maintain?

### ✅ Answer

For each active user, the system needs their latest:

`h_t`

So conceptually:

```text
User ID
→ Latest Hidden State
```

The model weights can be shared globally.

But hidden state is:

> per sequence/user.

---

## Question 2

Why is this not simply a machine-learning problem anymore?

### ✅ Answer

Stateful inference creates distributed-systems concerns.

The system must manage:

* state storage
* state lookup
* partitioning
* failure recovery
* user routing
* concurrent updates
* state expiration

So model architecture affects:

> infrastructure design.

---

## Question 3

Two transactions for the same user arrive simultaneously at different servers.

What could go wrong?

### ✅ Answer

Both servers might read the same old:

`h_t`

and independently calculate different:

`h_{t+1}`

states.

That creates:

> race conditions / inconsistent state updates.

The system needs ordering or concurrency control for a user's sequence.

---

## Question 4

Why does transaction ordering matter?

### ✅ Answer

RNN state evolves sequentially:

```text
h_t → h_{t+1} → h_{t+2}
```

Processing events out of order changes the hidden-state trajectory.

So:

```text
Transaction A → Transaction B
```

is not generally equivalent to:

```text
Transaction B → Transaction A
```

---

## Question 5

Would vanilla RNN automatically be your final production choice?

### ✅ Answer

No.

I would evaluate:

* dependency horizon
* accuracy
* latency
* state-management overhead
* memory
* training stability
* model size
* event ordering
* hardware
* drift

If very long-range behavior matters, I would compare:

* LSTM
* GRU
* attention-based alternatives
* other temporal architectures

on the real workload.

---

# 🎯 Final Applied Exercise

You are given:

```text
Embedding size D = 128
Hidden size H = 256
Output classes = 4
Sequence length = 1000
```

### Questions

1. What is `W_xh.shape`?
2. What is `W_hh.shape`?
3. What is `b_h.shape`?
4. What is the recurrent parameter count?
5. Does changing sequence length from `1000` to `2000` change that parameter count?
6. What does increase?
7. If only the final hidden state is used, what kind of task structure is this likely to represent?

### ✅ Answer

### 1.

`W_xh = 256 × 128`

### 2.

`W_hh = 256 × 256`

### 3.

`b_h = 256`

### 4. Recurrent Parameters

`256×128 + 256×256 + 256`

`= 32,768 + 65,536 + 256`

# `= 98,560`

### 5.

No.

Parameter count stays:

`98,560`

### 6.

The number of recurrent computations increases.

Therefore:

> compute/time increases.

### 7.

If the sequence produces one final output:

> **Many-to-One**

---

# 🧠 Final Assignment Mental Model

```text
Shared Model
↓
Same Recurrent Rule Across Time
↓
Different Inputs + Different Previous States
↓
Different Hidden States
↓
Sequence-Specific Context
```

And for production:

```text
Shared Weights
+
Per-Sequence State
+
Correct Event Ordering
=
Stateful RNN Inference
```

---

# ⭐ Golden Rule

> **The best way to understand an RNN is not to memorize its definition, but to trace what changes, what stays shared, what information is carried forward, and what breaks when the sequence becomes long or state is managed incorrectly.**
