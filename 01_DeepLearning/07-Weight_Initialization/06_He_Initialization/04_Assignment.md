# 💡 Assignment — He Initialization

> **Focus:** Understand why He Initialization is designed for ReLU, how `fan_in` and `fan_out` affect forward/backward variance, and how He differs from Xavier.

---

## 📌 Assignment 1 — Scenario Based

### 🧠 Scenario A — ReLU Network

You have a neural network layer:

Input Neurons = 100  
Output Neurons = 50  
Activation = ReLU

### Questions

### 1️⃣ What are `fan_in` and `fan_out`?

✅ **Answer**

fan_in = 100  
fan_out = 50

---

### 2️⃣ How many weights are in this fully connected layer?

✅ **Answer**

Total Weights  
=  
fan_in × fan_out

So:

100 × 50  
=  
5000

Therefore:

> **The layer has 5000 weights.**

---

### 3️⃣ What is the common He variance for forward preservation?

✅ **Answer**

Formula:

Var(W)  
=  
2 / fan_in

So:

Var(W)  
=  
2 / 100

=  
0.02

---

### 4️⃣ What is the standard deviation for He Normal?

✅ **Answer**

Std(W)  
=  
sqrt(2 / fan_in)

So:

Std(W)  
=  
sqrt(0.02)

≈  
0.141

---

## 🧠 Scenario B — Why the Factor `2`?

A developer asks:

> "Why don't we simply use `1 / fan_in` for a ReLU network?"

### 1️⃣ What would you explain?

✅ **Answer**

ReLU sets negative activations to zero.

For a roughly symmetric input distribution:

Approximately 50% positive  
Approximately 50% negative

After ReLU:

Positive values → Kept  
Negative values → 0

So part of the signal is removed.

He compensates by increasing the variance.

Without ReLU:

Var(W) ≈ 1 / fan_in

With ReLU compensation:

Var(W) ≈ 2 / fan_in

---

### 2️⃣ What is the core intuition?

✅ **Answer**

ReLU  
↓  
Removes part of the signal  
↓  
Need larger initial variance  
↓  
He Initialization  
↓  
2 / fan_in

---

## 📈 Scenario C — Forward vs Backward He

You have:

fan_in = 100  
fan_out = 500

### 1️⃣ Calculate the forward-preserving He variance.

✅ **Answer**

Var(W)  
=  
2 / fan_in

=  
2 / 100

=  
0.02

---

### 2️⃣ Calculate the backward-preserving He variance.

✅ **Answer**

Var(W)  
=  
2 / fan_out

=  
2 / 500

=  
0.004

---

### 3️⃣ Are they the same?

❌ **Answer: No**

Because:

fan_in ≠ fan_out

the forward and backward variance-preserving conditions are different.

---

### 4️⃣ What does `2 / fan_in` primarily preserve?

✅ **Answer**

It primarily preserves:

> **Forward activation variance**

---

### 5️⃣ What does `2 / fan_out` primarily preserve?

✅ **Answer**

It primarily preserves:

> **Backward gradient variance**

---

## ⚖️ Scenario D — Equal Layer Widths

Suppose:

fan_in = 256  
fan_out = 256

### 1️⃣ What is the forward He variance?

✅ **Answer**

2 / 256  
=  
0.0078125

---

### 2️⃣ What is the backward He variance?

✅ **Answer**

2 / 256  
=  
0.0078125

---

### 3️⃣ What does this tell us?

✅ **Answer**

When:

fan_in = fan_out

the forward and backward requirements become the same.

So He Initialization can approximately preserve both activation and gradient variance under the simplified assumptions.

---

## 🎲 Scenario E — He Normal vs He Uniform

Suppose:

fan_in = 100

### 1️⃣ What is He Normal standard deviation?

✅ **Answer**

Std(W)  
=  
sqrt(2 / 100)

≈  
0.141

So:

W ~ Normal(mean = 0, std ≈ 0.141)

---

### 2️⃣ What is the He Uniform range?

✅ **Answer**

Formula:

a  
=  
sqrt(6 / fan_in)

So:

a  
=  
sqrt(6 / 100)

≈  
0.245

Therefore:

W ~ Uniform(-0.245, +0.245)

---

### 3️⃣ Do He Normal and He Uniform have the same goal?

✅ **Answer**

Yes.

They use different distributions, but both aim for the same ReLU-aware variance scale.

---

## 🔵 Scenario F — Xavier or He?

You are given two networks.

### Network A

Activation = Tanh

### Network B

Activation = ReLU

### 1️⃣ Which initialization would you choose for Network A?

✅ **Answer**

**Xavier Initialization**

Because Xavier is commonly suited to `tanh` and sigmoid-style activations.

---

### 2️⃣ Which initialization would you choose for Network B?

✅ **Answer**

**He Initialization**

Because He is designed specifically for ReLU-family activations.

---

### 3️⃣ Why is He variance generally larger than Xavier for equal-width layers?

✅ **Answer**

Suppose:

fan_in = fan_out = n

Xavier:

Var(W)  
=  
2 / (n + n)

=  
1 / n

He:

Var(W)  
=  
2 / n

So He variance is about twice Xavier variance.

This compensates for ReLU zeroing negative activations.

---

# 📌 Assignment 2 — True / False

| # | Statement | Answer |
|---|---|---|
| 1 | He Initialization is also called Kaiming Initialization. | ✅ True |
| 2 | He Initialization is primarily designed for ReLU-family activations. | ✅ True |
| 3 | The standard He forward variance is `2 / fan_in`. | ✅ True |
| 4 | `2 / fan_in` means every weight has the value `2 / fan_in`. | ❌ False |
| 5 | ReLU helps motivate the factor `2` because negative activations are zeroed. | ✅ True |
| 6 | `fan_in × fan_out` gives the total number of weights in a dense layer. | ✅ True |
| 7 | `2 / fan_out` can be used to focus on backward gradient variance. | ✅ True |
| 8 | `fan_in` and `fan_out` always give the same He variance. | ❌ False |
| 9 | He completely guarantees that gradients can never vanish or explode. | ❌ False |
| 10 | He Normal and He Uniform use different distributions but similar variance goals. | ✅ True |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are designing a deep ReLU network.

One layer has:

fan_in = 128  
fan_out = 1024

A team member says:

> "We are using He Initialization with `2 / fan_in`, so both forward activations and backward gradients are guaranteed to stay perfectly stable."

### 1️⃣ Do you agree?

❌ **Answer: No**

Using:

Var(W)  
=  
2 / fan_in

gives:

2 / 128  
≈  
0.015625

This primarily targets **forward activation variance**.

---

### 2️⃣ What would the backward-preserving variance be?

✅ **Answer**

Var(W)  
=  
2 / fan_out

=  
2 / 1024

≈  
0.001953

This is much smaller.

---

### 3️⃣ Why are these values different?

✅ **Answer**

Because:

fan_in ≠ fan_out

Forward propagation depends on incoming connectivity.

Backward propagation depends on outgoing connectivity.

So one variance cannot perfectly preserve both directions when the layer widths are very different.

---

### 4️⃣ What would you explain to the team?

✅ **Answer**

I would explain:

fan_in  
↓  
Forward activation preservation

fan_out  
↓  
Backward gradient preservation

When:

fan_in ≈ fan_out

both requirements are similar.

When:

fan_in and fan_out differ significantly

there is a tradeoff.

---

### 5️⃣ Does this mean He Initialization is not useful?

❌ **Answer: No**

He Initialization is still highly useful.

It provides a principled ReLU-aware starting scale and helps reduce poor signal propagation.

The key is:

> **Initialization improves the starting conditions; it does not guarantee perfect variance preservation throughout training.**

---

# 🎯 Final Challenge

Complete the following flow:

ReLU  
↓  
Negative Activations → __________  
↓  
Part of Signal Is Lost  
↓  
Need Compensation  
↓  
Forward He Variance = __________  
Backward He Variance = __________

### ✅ Answer

ReLU  
↓  
Negative Activations → 0  
↓  
Part of Signal Is Lost  
↓  
Need Compensation  
↓  
Forward He Variance = 2 / fan_in  
Backward He Variance = 2 / fan_out

---

# 🧠 Final Mental Model

He Initialization  
↓  
ReLU-Aware Scaling

Forward:

fan_in  
↓  
2 / fan_in  
↓  
Preserve Activation Scale

Backward:

fan_out  
↓  
2 / fan_out  
↓  
Preserve Gradient Scale

If:

fan_in ≈ fan_out

then:

Forward and Backward Requirements  
≈  
Similar

---

# ⭐ Golden Rule

> **He Initialization compensates for ReLU's gating behavior. Use `fan_in` to reason about forward activations, `fan_out` to reason about backward gradients, and remember that the factor `2` comes from ReLU zeroing a large fraction of the signal.**