# 💡 AdamW Optimizer — Assignment

> **Objective:** Apply the concepts of AdamW to real engineering scenarios and reason from first principles.

---

# 🎯 Assignment 1 — Choosing the Right Optimizer

## Scenario A

You're training a 7B-parameter LLM from scratch.

Training loss decreases rapidly, but validation loss starts increasing after a few epochs.

### Solution

I would choose **AdamW**.

This is the classical limitation of Adam.

Adam optimizes the training loss very efficiently and converges quickly. However, it may allow the weights to grow unnecessarily large if doing so continues reducing the training loss. Large weights often make the model memorize the training data instead of learning the underlying patterns, resulting in poor generalization.

AdamW addresses this problem by introducing **Decoupled Weight Decay**, which independently shrinks the weights during every update while preserving Adam's optimization capabilities.

---

## Scenario B

You're training a small CNN on a clean image classification dataset.

Generalization is more important than fast convergence.

### Solution

I would choose **SGD with Momentum**.

On relatively small and clean datasets, SGD with Momentum generally produces better generalization than adaptive optimizers.

Although Adam-based optimizers usually converge faster, SGD often finds flatter minima that perform better on unseen data.

The trade-off is that SGD usually requires more epochs and careful learning-rate scheduling.

---

## Scenario C

You're fine-tuning a large Transformer on a small domain-specific dataset.

Overfitting becomes noticeable after a few thousand iterations.

### Solution

I would choose **AdamW**.

Fine-tuning on small datasets often leads to overfitting.

AdamW preserves Adam's fast optimization while preventing unnecessary weight growth through decoupled weight decay.

This keeps the model simpler and improves generalization on unseen samples.

---

## Scenario D

You're training a recommendation system with millions of sparse features.

### Solution

I would choose **AdaGrad**.

Recommendation systems are highly sparse because many users interact with only a tiny subset of available items.

AdaGrad automatically assigns larger learning rates to parameters that receive infrequent updates while reducing the learning rates for frequently updated parameters.

This makes AdaGrad particularly effective for sparse feature spaces.

---

# 🎯 Assignment 2 — Reasoning

## Q1. Why doesn't minimizing training loss always produce the best model?

### Solution

Minimizing the training loss alone may cause the model to overfit.

Instead of learning the underlying patterns, the model begins memorizing the training examples—including noise.

As a result:

- Training loss becomes very small.
- Validation and test loss start increasing.
- The model performs poorly on unseen data.

---

## Q2. Why is Weight Decay considered a form of regularization?

### Solution

Weight Decay continuously shrinks the model's weights during training.

Smaller weights generally produce smoother decision boundaries and simpler models.

This reduces overfitting and improves the model's ability to generalize to unseen data.

---

## Q3. Why is Decoupled Weight Decay mathematically cleaner than adding L2 Regularization to Adam?

### Solution

Adam rescales every parameter update using the EMA of historical squared gradients.

If L2 regularization is added directly to the gradient, that regularization term is also rescaled by Adam's adaptive mechanism.

As a result, different parameters receive different amounts of regularization depending on their gradient history.

AdamW avoids this problem by applying weight decay **after** the Adam update.

Therefore, every parameter receives consistent regularization independent of its adaptive learning rate.

---

## Q4. Why does AdamW usually generalize better than Adam?

### Solution

Adam focuses entirely on minimizing the loss.

If increasing certain weights continues reducing the training loss, Adam has no mechanism to prevent those weights from becoming unnecessarily large.

Large weights often lead to:

- Overfitting
- Complex decision boundaries
- High sensitivity to small input changes
- Poor generalization

AdamW solves this by introducing **Decoupled Weight Decay**, which continuously keeps the weights under control while Adam focuses on optimization.

---

# 🎯 Assignment 3 — Think Like an Optimizer Designer

Imagine AdamW had never been invented.

How would you improve Adam?

### Solution

### Step 1 — Observe the Problem

Adam learns very quickly and achieves low training loss.

However, it sometimes performs poorly on unseen data.

This suggests the model is overfitting rather than learning general patterns.

---

### Step 2 — Form a Hypothesis

Overfitting is often associated with unnecessarily large weights.

If the weights could be kept smaller during training, the model might generalize better.

---

### Step 3 — First Attempt

The obvious solution is to add L2 Regularization:

```text
L_new = L_original + λ Σ w²
```

This penalizes large weights by incorporating the penalty directly into the loss function.

---

### Step 4 — Discover the Limitation

Adam does not use the raw gradient directly.

Instead, it rescales every update using the EMA of historical squared gradients.

Therefore, the L2 penalty also gets rescaled.

This causes different parameters to receive different amounts of regularization.

But regularization should depend only on the weight size—not on the parameter's gradient history.

---

### Step 5 — The Key Insight

Learning from data and shrinking weights are two completely different objectives.

They should not share the same update pipeline.

---

### Step 6 — Final Solution

First perform the normal Adam update using only the true gradient:

```text
w ← w − η · AdamUpdate(∇L)
```

Then independently apply Weight Decay:

```text
w ← w − ηλw
```

The complete update becomes:

```text
w ← w − η · AdamUpdate(∇L) − ηλw
```

Now every parameter is regularized consistently, regardless of its adaptive learning rate.

This is exactly the core idea behind **AdamW**.

---

# 🎯 Assignment 4 — True or False

Explain your reasoning.

| Statement | Answer |
|-----------|--------|
| AdamW changes the gradient computation. | ❌ False |
| Weight Decay reduces the learning rate. | ❌ False |
| AdamW performs optimization and regularization independently. | ✅ True |
| Large weights always improve model accuracy. | ❌ False |
| AdamW is a completely new optimization algorithm. | ❌ False |

---

# 🎯 Assignment 5 — Design an Ideal Optimizer

List the properties an ideal optimizer should possess.

### Solution

An ideal optimizer should:

- ✅ Find the best direction for optimization.
- ✅ Determine an appropriate step size for every parameter.
- ✅ Prevent unnecessary weight growth to improve generalization.
- ✅ Correct initialization bias introduced by Exponential Moving Averages.

---

# 🚀 Staff Engineering Challenge

Your organization trains multiple AI systems.

Recommend the most suitable optimizer for each workload.

| Workload | Recommended Optimizer | Reason |
|----------|-----------------------|--------|
| Image Classification | **SGD + Momentum + Weight Decay** | Excellent generalization, stable optimization, widely used for CNNs. |
| Recommendation System | **AdaGrad (or SparseAdam)** | Handles sparse features by assigning larger learning rates to rarely updated parameters. |
| Large Language Models | **AdamW** | Fast convergence, adaptive learning rates, decoupled weight decay, industry standard for Transformers. |
| Robotics Control | **SGD + Momentum** (or carefully tuned Adam with Gradient Clipping) | Prioritizes stable and predictable optimization under noisy gradients. |
| Time-Series Forecasting | **Adam / AdamW** | Adaptive optimization handles complex sequential models; AdamW is preferred when overfitting is a concern. |

---

# ✅ Completion Checklist

- [ ] Selected the appropriate optimizer for different real-world scenarios.
- [ ] Understood why Adam required another improvement.
- [ ] Explained why Weight Decay improves generalization.
- [ ] Understood why Adam + L2 is not equivalent to AdamW.
- [ ] Derived AdamW from first principles.
- [ ] Designed the characteristics of an ideal optimizer.
- [ ] Completed the Staff Engineering Challenge.

---

# 🎯 Success Criteria

You are ready to move to the next lecture if you can confidently explain:

- Why Adam sometimes generalizes poorly.
- Why large weights lead to overfitting.
- Why AdamW separates optimization from regularization.
- Why Decoupled Weight Decay is mathematically cleaner than L2 Regularization.
- Why AdamW has become the default optimizer for modern Transformer and LLM training.