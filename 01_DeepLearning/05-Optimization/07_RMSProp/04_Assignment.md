# RMSProp — Assignment

---

# Assignment 1 — Choosing the Right Optimizer

You are given three different training scenarios.

## Scenario A

A neural network is trained on a relatively small dataset.

Training completes in only **50 iterations**.

The learning rate never becomes excessively small.

### Question

Which optimizer would you choose?

Explain your reasoning.

### Solution

For a small dataset, the computational cost of Batch Gradient Descent is not a major concern because each weight update processes only a limited number of samples.

Therefore, **Batch Gradient Descent** is a reasonable choice.

If faster convergence is desired while still benefiting from previous optimization history, **SGD with Momentum** is also a good option.

Adaptive optimizers such as RMSProp are unnecessary because the primary limitation they solve (learning rate decay over very long training runs) does not exist in this scenario.

---

## Scenario B

A model trains for **5 million iterations**.

Initially, optimization progresses normally.

After several million iterations, learning becomes extremely slow because the effective learning rate approaches zero.

### Question

Which optimizer would you recommend?

Why?

### Solution

This behaviour is characteristic of **AdaGrad**.

AdaGrad continuously accumulates squared gradients throughout training.

As a result, the accumulated history keeps increasing, causing the effective learning rate to become smaller and smaller.

Eventually, weight updates become almost negligible.

**RMSProp** addresses this limitation by replacing permanent memory with a weighted running average (EMA), allowing recent gradients to influence the learning rate more than very old gradients.

---

## Scenario C

Customer behaviour changes frequently because of:

* seasonal trends,
* festivals,
* product launches,
* promotions.

### Question

Which optimizer is more suitable?

Explain your reasoning.

### Solution

This problem requires an optimizer that adapts quickly to changing data distributions.

Using the complete historical gradient (AdaGrad) causes old customer behaviour to dominate future learning.

RMSProp gradually reduces the influence of older gradients while emphasizing recent gradients.

Therefore, it adapts much more effectively to continuously changing user behaviour.

---

# Assignment 2 — Effect of β

Suppose RMSProp uses

```text
β = 0.99
```

The underlying data distribution suddenly changes.

### Questions

* How quickly will the optimizer adapt?
* How does β influence historical information?
* What happens to training stability?

### Solution

A large β means historical information receives much greater importance than recent gradients.

Only a small fraction of the current gradient contributes to updating the running average.

As a result:

* adaptation becomes slower,
* historical behaviour dominates longer,
* optimization becomes more stable,
* but the model reacts more slowly to sudden changes.

Unlike AdaGrad, however, historical information still fades gradually because RMSProp uses an Exponential Moving Average rather than permanent accumulation.

---

# Assignment 3 — Debugging Training

Training log:

```text
Epoch 1
Loss ↓

Epoch 2
Loss ↓

...

Epoch 50
Loss ↓

...

Epoch 5000
Loss almost constant

Gradient is still non-zero.
```

### Questions

1. What optimizer-related problem do you suspect?
2. Which optimizer is most likely being used?
3. How would you verify your hypothesis?

### Solution

Several possibilities should be considered first.

* The effective learning rate may have become extremely small.
* Gradients may have become very small.
* The product of learning rate and gradient may now produce negligible updates.

The important clue is that optimization initially progresses well before slowing dramatically after a very large number of iterations.

This behaviour strongly suggests AdaGrad, where the accumulated squared gradients continue increasing throughout training, causing the effective learning rate to approach zero.

Verification can be performed by monitoring the effective learning rate or accumulated gradient history during training.

---

# Assignment 4 — Compare Optimizers

Complete the following comparison.

| Optimizer        | What is Remembered?              | Why?                                                                          | Limitation                                                                                         |
| ---------------- | -------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Gradient Descent | Nothing                          | Uses only the current gradient for each update.                               | No historical information is used.                                                                 |
| Momentum         | Previous velocity (movement)     | Accelerates optimization and reduces oscillations.                            | Uses one global learning rate for all parameters.                                                  |
| AdaGrad          | All historical squared gradients | Assigns adaptive learning rates to each parameter.                            | Treats every historical gradient equally, causing learning rates to shrink continuously.           |
| RMSProp          | Recent squared gradients (EMA)   | Maintains adaptive learning rates while gradually forgetting older gradients. | Older information gradually loses influence, making long-term recurring patterns harder to retain. |

---

# Assignment 5 — Choosing β

Two optimizers are proposed.

## Optimizer A

```text
EMA = 0.30 × Old + 0.70 × New
```

## Optimizer B

```text
EMA = 0.99 × Old + 0.01 × New
```

### Questions

* Which optimizer adapts faster?
* Which optimizer is more stable?
* Where might each fail?

### Solution

**Optimizer A**

* Gives much greater importance to recent gradients.
* Adapts quickly to changing data.
* Less stable because historical information disappears rapidly.

**Optimizer B**

* Strongly remembers historical gradients.
* Produces smoother optimization.
* Adapts slowly when the data distribution changes.

This illustrates the trade-off controlled by β between responsiveness and stability.

---

# Assignment 6 — Selecting an Optimizer for Real Applications

Choose the most appropriate optimizer for each application.

Explain your reasoning.

| Application             | Recommended Optimizer | Reason                                                                                                                   |
| ----------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Stock Market Prediction | RMSProp               | Market behaviour changes frequently. Recent information is more valuable than distant history.                           |
| Weather Prediction      | RMSProp / Momentum    | Weather patterns evolve gradually. Stable optimization with controlled adaptation is desirable.                          |
| Medical Diagnosis       | Momentum / RMSProp    | Stability and reliable convergence are generally preferred over sensitivity to noisy updates.                            |
| Fraud Detection         | AdaGrad / RMSProp     | Fraud-related features are often sparse. Adaptive learning rates help infrequently updated parameters learn effectively. |

---

# 🎯 Staff Engineer Challenge

Your company maintains **300 recommendation models**.

Business analysis identifies three categories.

| Model Type                               | Approximate Count |
| ---------------------------------------- | ----------------: |
| High-Velocity / Trend-Driven Models      |               120 |
| Sparse / Long-Tail Recommendation Models |               100 |
| Deep Hybrid Recommendation Models        |                80 |

### Questions

1. Which optimizer would you recommend for each category?
2. Justify every recommendation.
3. Explain why a single optimizer may not be optimal for every workload.

### Solution

| Model Category                 | Recommended Optimizer | Engineering Justification                                                                                                                                                                                 |
| ------------------------------ | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| High-Velocity / Trend-Driven   | RMSProp               | Customer behaviour changes continuously. RMSProp adapts quickly by emphasizing recent gradients while preventing long-term learning rate decay.                                                           |
| Sparse / Long-Tail             | AdaGrad               | Sparse features receive infrequent updates. AdaGrad preserves historically important updates and naturally assigns larger effective learning rates to rarely updated parameters.                          |
| Deep Hybrid (Two-Tower / DLRM) | Hybrid Strategy       | Sparse embedding layers benefit from AdaGrad-like behaviour, while dense neural components benefit from RMSProp-style adaptive optimization. Different model components may require different optimizers. |

### Engineering Takeaway

There is no universally best optimizer.

The appropriate optimizer depends on:

* data distribution,
* feature sparsity,
* training duration,
* rate of concept drift,
* computational constraints,
* and business objectives.

A Staff Engineer selects optimizers based on workload characteristics rather than personal preference.
