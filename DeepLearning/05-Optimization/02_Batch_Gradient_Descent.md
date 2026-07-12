# Batch Gradient Descent — Learning from the Entire Dataset

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- What is Batch Gradient Descent?
- Why was it invented?
- Why don't we update the weights after every training example?
- Why do we compute the gradient over the entire dataset?
- What are the advantages of Batch Gradient Descent?
- Why does it become impractical for modern deep learning?
- Why was Stochastic Gradient Descent eventually introduced?

---

# Chapter 1 — We Know How to Update Weights

In the previous lecture, we learned that the weight update equation is

$$
w = w - \eta \nabla L
$$

We now understand

- the Gradient tells us the direction,
- the Learning Rate tells us the step size.

But another question naturally arises.

```text
When should we perform this update?
```

Imagine we have a dataset containing

```text
100,000 images
```

Should we

- update after every image?

or

- wait until we've seen all 100,000 images?

This decision gave rise to different optimization strategies.

---

# Chapter 2 — A Simple Example

Suppose our dataset contains only four training examples.

```text
House 1

House 2

House 3

House 4
```

Our model makes predictions for each house.

Each prediction has its own error.

```text
House 1 → Error = +10

House 2 → Error = -4

House 3 → Error = +6

House 4 → Error = +2
```

Each example therefore contributes its own gradient.

```text
House 1 → Gradient = 8

House 2 → Gradient = -3

House 3 → Gradient = 5

House 4 → Gradient = 2
```

Now comes an important question.

Should we update the weights after seeing only the first house?

Or should we listen to **every training example** before making a decision?

---

# Chapter 3 — Listening to the Entire Class

Imagine you are a teacher.

Four students attempt the same mathematics question.

```text
Student A → Wrong

Student B → Correct

Student C → Slightly Wrong

Student D → Correct
```

Would you redesign your teaching strategy after hearing only Student A?

Probably not.

A better approach is to observe the performance of the **entire class**.

Only then do you decide what should be improved.

Batch Gradient Descent follows exactly the same philosophy.

Instead of reacting to one training example,

it first examines the **entire dataset**.

Only then does it update the weights.

---

# Chapter 4 — How Batch Gradient Descent Works

The algorithm is surprisingly simple.

Step 1

Run a Forward Pass on every training example.

```text
Image 1

↓

Prediction

↓

Loss
```

Repeat this for every image in the dataset.

---

Step 2

Compute the average loss (or equivalently, the average gradient).

Conceptually,

```text
Average Gradient

=

Gradient₁

+

Gradient₂

+

...

+

Gradientₙ

-----------------

n
```

---

Step 3

Update the weights **once**.

```text
All Data

↓

Average Gradient

↓

Single Weight Update
```

Only after considering every example do the model parameters change.

---

# Chapter 5 — Why Is This a Good Idea?

Since the update is based on the entire dataset,

the computed gradient is much more reliable.

Individual training examples may contain noise.

One example might suggest increasing a weight.

Another may suggest decreasing it.

By averaging over all examples,

these random fluctuations are reduced.

The resulting update is smoother and more stable.

Batch Gradient Descent therefore follows a very consistent path toward the minimum.

---

# Chapter 6 — The Biggest Problem

Everything sounds perfect.

So why doesn't everyone use Batch Gradient Descent?

Imagine training GPT.

Instead of four training examples,

suppose the dataset contains

```text
2 trillion tokens
```

Before updating the weights,

the optimizer would need to

- perform a Forward Pass over every token,
- compute every loss,
- compute every gradient,
- average all gradients.

Only then could it update the weights.

One update might take hours or even days.

Learning becomes painfully slow.

---

# Chapter 7 — Visualizing the Problem

Suppose we have one million training samples.

```text
Sample 1

↓

Sample 2

↓

...

↓

Sample 1,000,000

↓

One Weight Update
```

Notice something.

The model waits until it has processed **every single sample** before learning anything.

This delay makes Batch Gradient Descent impractical for modern deep learning.

---

# Chapter 8 — Looking Ahead

Batch Gradient Descent gives us

- accurate gradients,
- smooth optimization,
- stable learning.

However,

it is computationally expensive.

Researchers naturally asked:

> **Do we really need to wait for the entire dataset before updating the weights?**

What if we updated the weights after just **one training example**?

That question led to

> **Stochastic Gradient Descent (SGD)**

which we'll study in the next lecture.

---

# Interview Insight

A common interview question is:

> **Why is Batch Gradient Descent rarely used for training modern Large Language Models?**

A strong answer is:

Batch Gradient Descent computes gradients using the entire dataset before updating the weights.

Although this produces accurate and stable gradient estimates, it becomes computationally impractical for datasets containing millions or billions of training examples.

Modern deep learning instead uses Mini-Batch Gradient Descent, which balances computational efficiency with stable learning.

## 💡 Engineer's Insight

Batch Gradient Descent follows a simple philosophy:

> **One training example is rarely representative of the entire dataset.**

A single sample may be noisy, mislabeled, or simply unusual.

Instead of allowing one example to dictate how the model should learn, Batch Gradient Descent first observes the behavior of the **entire dataset**.

Only after considering all training examples does it update the weights.

This produces a more reliable estimate of the true direction in which the loss should decrease.

---
---

# 🧠 Engineer Discussion & Intuitive Q&A

## Q1 — When Are the Weights Updated?

Suppose your dataset contains

```text
10,000 training examples
```

Using **Batch Gradient Descent**, after how many training examples are the weights updated?

### Answer

Batch Gradient Descent updates the weights **only after processing the entire dataset**.

For every training example, the model performs:

```text
Forward Pass

↓

Prediction

↓

Loss Computation

↓

Gradient Computation
```

After all training examples have been processed, the gradients are aggregated (conceptually averaged), and a **single weight update** is performed.

This philosophy is based on the idea that one training example is rarely representative of the entire dataset.

By considering all examples together, Batch Gradient Descent produces more reliable gradient estimates, resulting in smoother optimization and more stable learning.

---

## Q2 — Why Don't We Learn From One Training Example?

Imagine you are teaching a class of 100 students.

One student performs poorly on an exam.

Should you immediately redesign your entire teaching strategy?

### Answer

No.

One student's performance may not represent the performance of the entire class.

There could be many reasons:

- The student was absent.
- The student did not prepare.
- The student misunderstood the question.

Instead, a teacher first observes the performance of the entire class before deciding whether the teaching strategy needs improvement.

Batch Gradient Descent follows exactly the same philosophy.

Rather than allowing one training example to determine the weight update, it first considers the behavior of the **entire dataset** before making a learning decision.

---

## Q3 — How Does Batch Gradient Descent Handle Noisy Data?

Suppose your dataset contains

```text
999 good samples

1 noisy sample
```

The noisy sample suggests moving the weights in the completely opposite direction.

Why is Batch Gradient Descent less affected by this noisy sample?

### Answer

Batch Gradient Descent computes the gradient using the entire dataset.

Although one noisy sample may suggest an incorrect direction, the remaining 999 samples collectively dominate the final gradient.

As a result, the influence of a single noisy sample becomes very small.

This makes Batch Gradient Descent more stable and less sensitive to outliers than algorithms that update weights after every training example.

---

## Q4 — Is Batch Gradient Descent Always Better?

Someone argues:

> "Batch Gradient Descent is always better because it uses more data."

Do you agree?

### Answer

Not necessarily.

Batch Gradient Descent produces accurate and stable gradient estimates because it considers the entire dataset.

However, before every weight update, it must:

```text
Process every training sample

↓

Forward Pass

↓

Loss Computation

↓

Gradient Computation

↓

Aggregate Gradients

↓

Single Weight Update
```

For modern datasets containing millions or billions of samples, this process becomes extremely time-consuming.

The model spends a long time computing before learning anything.

Although mathematically stable, Batch Gradient Descent is often computationally impractical for large-scale deep learning.

---

## Q5 — Can We Train GPT Using Pure Batch Gradient Descent?

Suppose you are training GPT using

```text
2 trillion tokens
```

Would pure Batch Gradient Descent be practical?

### Answer

No.

Before performing a single weight update, the optimizer would have to process the entire dataset.

That means:

```text
Forward Pass

↓

Loss Computation

↓

Gradient Computation

↓

Aggregate Gradients

↓

One Weight Update
```

With trillions of training tokens, one update could take an enormous amount of time.

The optimizer waits until the entire dataset has been processed before learning anything.

For this reason, pure Batch Gradient Descent is not practical for training modern Large Language Models.

---

## Q6 — What Happens When the Dataset Grows?

Suppose the dataset grows from

```text
1,000 samples

↓

100 million samples
```

How does this affect training?

### Answer

As the dataset grows:

- More computation is required before every weight update.
- The time required for one update increases significantly.
- During one epoch, the weights are still updated only once.

The main bottleneck is **computation time**, not the amount of memory required to store gradients.

In practice, gradients are accumulated while processing the dataset rather than storing every individual gradient separately.

---

## Q7 — If the Dataset Never Changes, Why Does the Gradient Change?

Suppose the dataset remains exactly the same across multiple epochs.

Will Batch Gradient Descent compute the same gradient every epoch?

### Answer

No.

Although the dataset remains unchanged, the **model weights are updated after every epoch**.

Since the model's predictions depend on its weights, changing the weights changes:

```text
Weights

↓

Predictions

↓

Loss

↓

Gradient
```

Therefore,

```text
Gradient₁ ≠ Gradient₂
```

even when the dataset is exactly the same.

---

## 💡 Engineer's Insight

Batch Gradient Descent follows a simple but powerful philosophy:

> **One training example is rarely representative of the entire dataset.**

A single sample may be noisy, mislabeled, or simply unusual.

Instead of allowing one example to dictate how the model should learn, Batch Gradient Descent first observes the behavior of the **entire dataset**.

Only then does it update the weights.

This produces a more reliable estimate of the true direction in which the loss should decrease.

---

# 🎯 Staff Engineer Challenge

Every night, your recommendation system receives

```text
500 million new user interactions.
```

A junior engineer proposes:

> "Let's retrain the recommendation model every night using pure Batch Gradient Descent over the entire dataset."

As the Technical Lead, how would you evaluate this proposal?

### Discussion

Before discussing optimization algorithms, a Staff Engineer first evaluates the problem from an engineering perspective.

Questions to investigate include:

### 1. Is the New Data Useful?

- Are these interactions genuine?
- Are they noisy, duplicated, or generated by bots?
- Will training on this data actually improve the model?

Poor-quality data can degrade model performance regardless of the optimization algorithm.

---

### 2. Does the Model Actually Need Retraining?

Before starting an expensive training job, evaluate the current model.

Questions include:

- Is model accuracy actually degrading?
- Are users experiencing worse recommendations?
- Is retraining justified by measurable business metrics?

Optimization should solve a real problem, not become a routine habit.

---

### 3. What Are the Infrastructure Costs?

Large-scale retraining requires significant resources.

Consider:

- GPU availability
- Training time
- Validation and testing effort
- Deployment time
- Infrastructure cost

Engineering decisions must balance model improvement with operational constraints.

---

### 4. Is Nightly Batch Gradient Descent Even Practical?

Every night introduces another

```text
500 million
```

new interactions.

If one Batch Gradient Descent training run takes many hours, the next day's data arrives before the previous training cycle finishes.

The system eventually falls behind.

Operational feasibility is just as important as algorithmic correctness.

---

### 5. Can We Think of a Better Training Strategy?

Rather than immediately retraining using the entire dataset, consider whether a more efficient strategy could achieve similar results.

This question naturally motivates the optimization techniques that we will study in the upcoming lectures.

---

## 💡 Staff Engineer Insight

A Staff Engineer does not begin by asking,

> "Which optimization algorithm should we use?"

Instead, they ask:

- Do we actually need retraining?
- Is the incoming data trustworthy?
- What business value will retraining provide?
- Can our infrastructure support this approach?
- Is there a simpler or more scalable solution?

Only after understanding the complete system do they select the most appropriate optimization strategy.

This mindset distinguishes an engineer who understands algorithms from one who can design and operate large-scale AI systems.


# Key Takeaways

- Batch Gradient Descent computes gradients using the **entire dataset**.
- Only one weight update is performed after processing all training examples.
- Averaging across all examples produces smooth and stable gradient estimates.
- Batch Gradient Descent is computationally expensive for large datasets.
- Its poor scalability motivated the development of Stochastic Gradient Descent.