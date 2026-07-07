# Part 5 — GELU (Session 2)
# From Bell Curves to GPT

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to answer:

- What is the Gaussian Distribution?
- What is a CDF?
- Why does GELU use a Gaussian CDF instead of a PDF?
- Why is the input multiplied by a probability?
- Why is GELU smooth?
- Why does GPT use an approximation instead of the exact formula?
- Why is GELU superior for Transformers?

---

# Chapter 50 — A Question That Changed Deep Learning

Last time we concluded that neurons shouldn't make hard decisions.

Instead of

```
Keep

Discard
```

they should ask

```
How likely am I to be useful?
```

Now comes the next question.

Suppose a neuron outputs

```
x = 2.7
```

How do we decide whether it deserves

```
40%

70%

95%
```

of the output?

Where does that percentage come from?

---

# Chapter 51 — Nature Loves Bell Curves

Imagine measuring

- Human height
- IQ scores
- Exam marks
- Manufacturing errors
- Sensor noise

Most values cluster around the average.

Very small values are rare.

Very large values are also rare.

If we plot these observations,

we obtain a familiar shape.

```
                *
             *     *
           *         *
         *             *
-------*-----------------*-------
```

This is the

# Gaussian Distribution

also called

# Normal Distribution.

---

# Chapter 52 — Why Should AI Care?

Suppose a neuron outputs

```
0.02
```

That's very close to zero.

It might just be noise.

Now suppose another neuron outputs

```
8.3
```

That's a very strong activation.

Intuitively,

we trust the second neuron much more.

Researchers wanted a mathematical way of expressing this intuition.

Small activations

↓

Less confidence

Large activations

↓

More confidence

The Gaussian distribution naturally models this gradual increase in confidence.

---

# Chapter 53 — PDF vs CDF

This is one of the most common interview questions.

There are two important functions associated with a Gaussian.

## 1. PDF (Probability Density Function)

Think of it as answering

> "How dense is the bell curve at exactly this point?"

```
              *
           *     *
         *         *
-------*-------------*-------
```

The PDF tells us the height of the curve.

But notice something.

It doesn't tell us

> "How much probability has accumulated so far?"

For GELU,

that's exactly what we need.

---

## 2. CDF (Cumulative Distribution Function)

CDF answers a different question.

> "What fraction of values lie to the left of x?"

Instead of measuring height,

it measures accumulated probability.

Its graph looks like this.

```
                ________

            ___/

        ___/

______/____________________
```

Notice something beautiful.

The output is always between

```
0

and

1
```

Exactly what we wanted.

A probability.

---

# Chapter 54 — Why CDF Is Perfect for GELU

Suppose

```
x = -5
```

Almost every Gaussian value lies to the right.

So

```
CDF(-5)

≈ 0
```

Now suppose

```
x = 5
```

Almost every value lies to the left.

```
CDF(5)

≈ 1
```

For values near zero,

the probability changes smoothly.

```
-5 ---- -2 ---- 0 ---- 2 ---- 5

↓

0      0.02   0.5   0.98   1
```

Exactly what we wanted.

Large activations receive probabilities close to

```
1
```

Small activations receive probabilities near

```
0.5
```

Negative activations gradually approach

```
0
```

No sudden jumps.

---

# Chapter 55 — Building GELU Step by Step

We now have

```
Neuron Output

↓

x
```

We also know

```
Confidence

↓

Φ(x)
```

where

```
Φ(x)
```

is the Gaussian CDF.

Now ask yourself.

Should the output become

```
Φ(x)
```

No.

Why?

Because we've lost the original information.

Suppose

```
x = 10
```

Its confidence is almost

```
1
```

Returning only

```
1
```

would throw away the magnitude.

Instead,

we want

```
Original Information

×

Confidence
```

which gives

```text
Output = x × Φ(x)
```

That is GELU.

---

# Chapter 56 — Understanding the Formula

```text
GELU(x) = x × Φ(x)
```

Break it into two parts.

```
x
```

The actual information produced by the neuron.

```
Φ(x)
```

How confident we are that this information deserves to survive.

The multiplication performs a soft filtering.

Not

```
Keep

Discard
```

Instead

```
Scale according to confidence.
```

---

# Chapter 57 — Why Not Use PDF?

Many interviewers love asking this.

Suppose we used

```
PDF(x)
```

instead.

The PDF is largest at

```
0
```

and decreases toward both sides.

That would imply

```
Strong activations

↓

Receive smaller weights
```

Exactly the opposite of what we want.

The CDF behaves naturally.

Larger activations receive larger confidence.

---

# Chapter 58 — But GPT Doesn't Compute This Exactly

Unfortunately,

computing the Gaussian CDF precisely is computationally expensive.

Remember,

GPT performs this calculation

billions of times

for every forward pass.

Researchers therefore found an approximation.

```text
GELU(x)

≈ 0.5x × [1 + tanh( √(2/π) × (x + 0.044715x³) )]
```

Don't panic.

You are **not expected to memorize** this equation.

The important idea is

> It behaves almost identically to the exact GELU,

while being much faster on GPUs.

Modern frameworks such as PyTorch implement this optimized approximation.

---

# Chapter 59 — Comparing ReLU and GELU

Consider three neuron outputs.

| Input | ReLU | GELU (Approximate Behaviour) |
|------:|-----:|-----------------------------:|
| -2 | 0 | Small negative value |
| 0 | 0 | 0 |
| 0.5 | 0.5 | ~0.35 |
| 2 | 2 | ~1.95 |
| 5 | 5 | ~5 |

Notice something.

ReLU always keeps positive values completely.

GELU asks

> "Is this activation strong enough?"

Weak positive activations are reduced.

Very strong activations pass almost unchanged.

Negative activations are not abruptly discarded.

Everything changes smoothly.

---

# Chapter 60 — Why Transformers Love GELU

Language is uncertain.

Every token contributes differently.

When the Feed Forward Network processes token representations,

not every feature deserves equal importance.

GELU softly suppresses weak features,

while allowing important semantic features to pass.

This smooth behaviour leads to

- Better optimization
- Better gradient flow
- Richer feature representations
- Improved language understanding

That's why GPT uses GELU inside every Feed Forward Network.

---

# 🎯 Interview Insight

If asked,

> Why does GPT use GELU instead of ReLU?

A strong answer is:

> "ReLU makes hard threshold-based decisions by either passing or discarding activations. Language understanding is inherently uncertain, where features often have varying degrees of importance. GELU performs probabilistic gating using the Gaussian CDF, allowing activations to pass proportionally to their confidence. This produces smoother optimization, richer feature representations, and empirically better performance in Transformer-based language models."

---

# 🧠 Common Interview Mistakes

❌ "GELU is just a smoother ReLU."

Better:

> "GELU changes the activation philosophy. Instead of thresholding activations, it scales them according to a probability derived from the Gaussian CDF."

---

❌ "GPT uses GELU because it's newer."

Better:

> "GPT uses GELU because probabilistic gating aligns better with the uncertainty and nuanced feature interactions found in language, whereas ReLU's hard threshold can discard potentially useful information."

---

# 📖 Key Takeaways

- Gaussian distributions model gradual confidence.
- CDF outputs values between 0 and 1, making it ideal for probabilistic gating.
- GELU multiplies the activation by its confidence.
- The CDF is used instead of the PDF because we need cumulative probability, not density.
- GPT uses a computational approximation of GELU for efficiency.
- GELU preserves subtle semantic information better than ReLU.

---

# 📖 End of Activation Functions Module

Congratulations!

We've now completed the evolutionary journey:

```
Step Function
        ↓
Could decide
Couldn't learn

Sigmoid
        ↓
Could learn
Vanishing gradients

ReLU
        ↓
Fast
Deep learning became practical

Leaky ReLU
        ↓
Reduced dead neurons

GELU
        ↓
Soft probabilistic gating
Perfect for Transformers
```

In the next module, we'll answer another fundamental question:

> **"Once a neural network makes a prediction, how does it know whether it was right or wrong?"**

That question leads us to **Loss Functions**.