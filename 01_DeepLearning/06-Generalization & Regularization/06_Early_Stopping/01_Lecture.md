# 📘 Early Stopping

> **Module:** Generalization & Regularization
>
> **Objective:** Understand why training a neural network for too long causes overfitting, how Early Stopping prevents it, and why it is one of the simplest yet most effective regularization techniques.

---

# 📖 Chapter 1 — Why Do Neural Networks Overfit?

In the previous lectures, we learned several techniques that reduce overfitting.

- L1 Regularization
- L2 Regularization
- Elastic Net

Notice something interesting.

All of them modify the **loss function**.

They change **what the optimizer tries to minimize.**

But here's an important question.

> **Can we reduce overfitting without changing the loss function at all?**

The answer is

**Yes.**

Sometimes the model is already good enough.

The only problem is that we continue training **after it has learned the important patterns.**

---

Imagine training a neural network.

```text
Epoch 1
    ↓
Model learns basic patterns

Epoch 10
    ↓
Model learns useful relationships

Epoch 20
    ↓
Model generalizes well

Epoch 40
    ↓
Starts memorizing training noise

Epoch 80
    ↓
Overfitting
```

The model initially learns genuine patterns.

But after sufficient training,

it starts memorizing noise present in the training data.

This memorization improves training accuracy,

but hurts performance on unseen data.

---

This gives us an interesting idea.

Instead of forcing the model to continue training,

why not stop at the point where it generalizes the best?

That idea is called

# **Early Stopping**

---

# 📖 Chapter 2 — The Intuition Behind Early Stopping

Think of preparing for an exam.

Initially,

every hour of study improves your understanding.

```text
Study
    ↓
Understanding increases
```

Eventually,

you understand almost everything.

If you continue studying the same questions repeatedly,

you're no longer learning concepts.

You're memorizing answers.

The same thing happens in neural networks.

Initially,

the model learns useful features.

Later,

it memorizes individual training examples.

Early Stopping prevents the model from entering this memorization phase.

---

Instead of asking

> "Has training loss become minimum?"

we ask

> "Has validation performance stopped improving?"

This small change dramatically improves generalization.

---

# 📖 Chapter 3 — Training Loss vs Validation Loss

To understand Early Stopping,

we must observe two different losses.

## Training Loss

Measures how well the model fits the training data.

## Validation Loss

Measures how well the model performs on unseen data.

---

During training,

both losses initially decrease.

```text
Epoch

↓

Training Loss ↓

Validation Loss ↓
```

The model is learning genuine patterns.

Everything looks good.

---

After some point,

something interesting happens.

Training loss continues decreasing.

But validation loss stops improving.

```text
Training Loss

\
 \
  \
   \
    \
     \

Validation Loss

\
 \
  \
   \
    \__
       \
        \
         \
```

The model is still improving on the training set,

but its performance on unseen data is getting worse.

This is the beginning of **overfitting.**

---

The best model is **not** the one with the lowest training loss.

The best model is the one with the **lowest validation loss.**

---

# 📖 Chapter 4 — The Core Idea of Early Stopping

The entire algorithm can be summarized in one sentence.

> **Stop training when validation performance stops improving.**

Notice something important.

We are **not** trying to minimize training loss forever.

Instead,

we monitor the validation metric.

Once it stops improving,

we terminate training.

---

Visually,

```text
Training Loss

Continues ↓

Validation Loss

↓

Improves

↓

Best Point ✅

↓

Starts Increasing

↓

Stop Training
```

The model saved at the **best validation performance** is used for inference.

Not the final model.

---

# 📖 Chapter 5 — Why Don't We Stop Immediately?

Suppose validation loss behaves like this.

```text
Epoch 10

Validation Loss = 0.45

Epoch 11

Validation Loss = 0.46

Epoch 12

Validation Loss = 0.44
```

If we stopped immediately at Epoch 11,

we would miss a better model at Epoch 12.

Validation metrics naturally fluctuate.

Small increases do not necessarily mean overfitting has begun.

Therefore,

we need some tolerance.

This introduces a new concept.

# **Patience**

---

# 📖 Chapter 6 — What is Patience?

Patience tells the model

> "Wait for a few epochs before giving up."

Example

```text
Patience = 3

Epoch 18 ✅ Best Model

Epoch 19 ❌ No Improvement

Epoch 20 ❌ No Improvement

Epoch 21 ❌ No Improvement

Stop Training
```

Here,

the training continues for three additional epochs.

If no improvement occurs,

training stops automatically.

---

Why is this useful?

Because validation loss is noisy.

Temporary fluctuations should not terminate training.

Patience avoids stopping too early.

---

# 📖 Chapter 7 — Early Stopping Algorithm

The overall workflow is simple.

```text
Initialize Training

        ↓

Train One Epoch

        ↓

Evaluate Validation Loss

        ↓

Improved?

 ┌───────────────┐
 │      Yes      │
 └──────┬────────┘
        │
Save Best Model
Reset Patience
        │
        ▼
Continue Training

──────────────

No

↓

Increase Patience Counter

↓

Patience Exceeded?

↓

Yes

↓

Stop Training

↓

Restore Best Model
```

Notice that

the model used after training is the **best saved model**,

not necessarily the model from the final epoch.

---

# 📖 Chapter 8 — Why Does Early Stopping Work?

Think back to the Bias-Variance Tradeoff.

```text
Very Few Epochs
        ↓
High Bias
(Underfitting)

Optimal Epoch
        ↓
Best Generalization

Too Many Epochs
        ↓
High Variance
(Overfitting)
```

Early Stopping automatically finds a point close to the optimal balance.

Instead of manually deciding

"Train for exactly 50 epochs",

we allow validation performance to decide when learning should stop.

---

# 📖 Chapter 9 — Advantages of Early Stopping

✅ Very easy to implement

No change to the model architecture.

---

✅ No change to the loss function

Unlike L1 or L2,

Early Stopping does not modify the optimization objective.

---

✅ Faster training

Training ends automatically once learning saturates.

No unnecessary epochs are executed.

---

✅ Better generalization

Prevents the model from memorizing training noise.

---

✅ Works with almost every neural network

- CNN
- RNN
- Transformer
- MLP

---

# 📖 Chapter 10 — Limitations of Early Stopping

❌ Requires a validation dataset

Without validation data,

the stopping criterion cannot be computed.

---

❌ Patience must be chosen carefully

Very small patience

→ Stops too early.

Very large patience

→ Allows overfitting.

---

❌ Sensitive to noisy validation metrics

Occasional fluctuations can make training difficult to interpret.

---

# 📖 Chapter 11 — Early Stopping vs L1/L2

Although all three reduce overfitting,

they work very differently.

| Technique | Main Idea |
|-----------|-----------|
| L1 | Removes unnecessary weights |
| L2 | Shrinks large weights |
| Elastic Net | Combines L1 and L2 |
| Early Stopping | Stops learning before memorization |

Notice that

Early Stopping does **not** change the model.

It changes **how long we train the model.**

---

# 📖 Chapter 12 — Key Takeaways

✅ Overfitting usually occurs after prolonged training.

✅ Training loss can continue improving while validation loss becomes worse.

✅ The best model is the one with the lowest validation loss.

✅ Early Stopping monitors validation performance instead of training loss.

✅ Patience prevents training from stopping because of temporary fluctuations.

✅ The best-performing checkpoint is restored after training.

✅ Early Stopping is one of the simplest and most widely used regularization techniques in deep learning.