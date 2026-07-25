# Lecture 6 — L1 Regularization
## Why Do Good Models Still Fail?

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to answer:

- Why do highly complex models overfit?
- Why isn't reducing epochs always the right solution?
- What is Regularization?
- Why do we intentionally add a penalty to the loss function?
- Why do we penalize weights instead of predictions?
- What problem was L1 Regularization trying to solve?
- How does L1 help improve generalization?

---

# 📖 Chapter 1 — We Finally Built a Powerful Neural Network

Let's continue our journey.

So far, we have learned:

- Hidden Layers made Neural Networks expressive.
- Activation Functions made them trainable.
- Gradient Descent optimized millions of parameters.
- Backpropagation allowed the model to learn from mistakes.

Everything seems perfect.

Suppose we build a Deep Neural Network for detecting fraudulent credit card transactions.

After several hours of training, we obtain the following results.

```text
Training Accuracy   : 99%

Validation Accuracy : 82%
```

At first glance, this looks impressive.

Many beginners immediately celebrate.

> "Our model has achieved 99% accuracy!"

The model is deployed to production.

A week later...

Customers begin reporting fraudulent transactions that the model completely missed.

The production accuracy drops to

```text
79%
```

Everyone is confused.

The model learned.

Gradient Descent worked.

Backpropagation worked.

The loss decreased.

So why did the model still fail?

---

# 🤔 A Bigger Question

Until now, every lecture focused on one goal:

> How do we make Neural Networks learn?

But today we must ask a different question.

> **How do we stop Neural Networks from learning too much?**

Surprisingly,

this turns out to be just as important.

---

# 📖 Chapter 2 — Learning vs Memorizing

Imagine two students preparing for an exam.

### Student A

- Understands the concepts.
- Solves different types of questions.
- Performs well even when the paper changes.

### Student B

- Memorizes every previous year's paper.
- Scores perfectly on practice tests.
- Panics when the exam asks something new.

Who is more likely to succeed?

Obviously,

Student A.

Machine Learning models behave exactly the same way.

A good model should learn

```text
Patterns

↓

Relationships

↓

Concepts
```

It should **not** memorize

```text
Noise

↓

Outliers

↓

Accidental Correlations
```

Unfortunately,

high-capacity neural networks are extremely good at memorization.

---

# 📖 Chapter 3 — The Real Culprit

Let's revisit the Bias–Variance Tradeoff.

We learned that highly complex models have

- Low Bias
- High Variance

That means they fit the training data almost perfectly.

But how exactly do they achieve this?

Let's imagine a House Price Prediction model.

Available features are

```text
Area

Bedrooms

Age

Location

Parking

Garden

Wall Color

Owner's Lucky Number
```

Clearly,

features like

```text
Area

Location

Bedrooms
```

are meaningful.

But what about

```text
Wall Color

Owner's Lucky Number
```

These features should have little or no influence on the house price.

Yet,

a highly flexible model may still assign them large weights if doing so slightly reduces the training error.

The model begins to rely on information that has no real predictive value.

This is exactly how overfitting begins.

---

# 💡 First Principle

A Neural Network does not know which features are genuinely important.

It only knows one thing:

> Reduce the training loss.

If assigning a larger weight to a feature appears to reduce the training loss—even if that relationship exists only because of noise, sampling effects, or accidental correlations in the training data—the optimizer will happily increase that weight.

The optimizer has no concept of

- common sense,
- causality,
- or feature importance.

It simply follows the mathematics.

---

## 🤔 But How Can a Useless Feature Reduce the Loss?

Imagine we are building a **House Price Prediction** model.

Our dataset contains only **20 houses**.

The available features are

```text
Area

Bedrooms

Location

House Number
```

Clearly,

```text
Area

Bedrooms

Location
```

can influence the price.

But **House Number** has no logical relationship with the value of a house.

Now suppose our small training dataset happens to look like this.

```text
House Number      Price

101               ₹2.1 Cr

102               ₹2.0 Cr

103               ₹2.2 Cr

104               ₹2.3 Cr

105               ₹2.1 Cr
```

Notice something.

Every expensive house in this small dataset happens to have a higher house number.

This is **not a real relationship.**

It is simply a coincidence in the training data.

However,

the optimizer doesn't know that.

It only observes

```text
Higher House Number

↓

Higher House Price
```

During training it discovers

> "If I increase the weight associated with House Number, the training loss decreases."

Mathematically,

it is correct.

The training loss really does decrease.

But the model has not discovered a genuine pattern.

It has merely learned an **accidental correlation** that exists only in this small training dataset.

Now imagine the model is deployed.

The very first new house it encounters is

```text
House Number = 220

Price = ₹75 Lakh
```

The model predicts

```text
Very Expensive
```

because it learned

```text
Higher House Number

↓

Higher House Price
```

The prediction is completely wrong.

Not because Gradient Descent failed.

Not because Backpropagation failed.

But because the model memorized a coincidence instead of learning the true relationship.

This phenomenon is called **overfitting**.

---

## 💡 Engineering Insight

The optimizer cannot distinguish between

```text
A Genuine Pattern

and

An Accidental Correlation.
```

If both reduce the training loss,

the optimizer treats them as equally valuable.

Our challenge is to stop the model from becoming **too confident about patterns that exist only by chance.**

That is exactly what **Regularization** is designed to do.

---

# 📖 Chapter 4 — Thinking Like an Engineer

Imagine your company hires 100 consultants to help make a business decision.

Some consultants are experts.

Some give average advice.

Some contribute nothing useful.

Some even create confusion.

Now imagine paying all of them equally.

The final decision becomes noisy.

Instead,

suppose you gradually remove everyone who adds little value.

Eventually,

only the truly useful experts remain.

The decision becomes

- simpler,
- cleaner,
- and more reliable.

This is exactly the idea behind **Regularization**.

Instead of allowing every weight to become large,

we encourage the model to keep only the weights that truly matter.

---

# 📖 Chapter 5 — The Birth of Regularization

Researchers realized something profound.

Until now,

our optimizer had only one objective.

```text
Minimize Prediction Error
```

Nothing else mattered.

If memorizing the training data reduced the loss,

the optimizer considered that a success.

But from an engineering perspective,

this is not what we actually want.

We want two things at the same time.

```text
Good Predictions

+

Simple Models
```

In other words,

the optimizer should not only minimize prediction error,

it should also discourage unnecessary complexity.

This idea became known as

# Regularization.

---

# 📖 Chapter 6 — A New Optimization Objective

Instead of minimizing only

```text
Prediction Loss
```

researchers proposed adding an additional cost.

Every time the model creates unnecessarily large weights,

it should pay a penalty.

The new objective becomes

```text
Total Loss

=

Prediction Loss

+

Penalty
```

Notice something important.

We are **not** changing the dataset.

We are **not** changing Gradient Descent.

We are **not** changing Backpropagation.

We are simply changing **what the optimizer considers a "good" solution.**

The optimizer must now balance two competing goals:

```text
Fit the Training Data

AND

Keep the Model Simple
```

This single idea changed modern Machine Learning.

---

# 🎯 Key Takeaways

- High-capacity models can memorize instead of generalize.
- Overfitting occurs when the model learns noise and accidental patterns.
- Gradient Descent only minimizes the objective we provide.
- Without constraints, the optimizer happily creates unnecessarily complex models.
- Regularization introduces a penalty that encourages simpler solutions.
- The optimizer must now balance prediction accuracy with model complexity.

---

# 🤔 Curiosity Corner

We now know that Regularization adds a penalty.

But a new question naturally appears.

> **What exactly should we penalize?**

Should we penalize

- predictions?
- neurons?
- gradients?
- or something else?

Researchers discovered that the answer lies in one of the most important quantities inside every Neural Network...

**its weights.**

# 📖 Chapter 7 — Why Penalize the Weights?

Let's continue.

At the end of the previous chapter, we arrived at a new optimization objective.

```text
Total Loss

=

Prediction Loss

+

Penalty
```

But this immediately raises an important question.

> **What exactly should the penalty depend on?**

Should we penalize

- the predictions?
- the neurons?
- the gradients?
- the dataset?

Researchers realized that none of these are the real problem.

The real problem lies elsewhere.

The model becomes complex because of its **weights**.

Large weights mean the model is relying heavily on certain features.

The more extreme these weights become,

the more likely the model is memorizing the training data instead of learning general patterns.

Therefore,

instead of penalizing predictions,

we penalize the weights themselves.

---

# 💡 First Principle

Remember,

every prediction made by a neural network ultimately depends on its weights.

```text
Input

↓

Weights

↓

Weighted Sum

↓

Activation

↓

Prediction
```

If we can control the weights,

we can control the complexity of the model.

---

# 📖 Chapter 8 — The Idea Behind L1 Regularization

Imagine you are packing for a trip.

Your airline allows only **20 kg** of luggage.

Every unnecessary item increases the weight.

Very quickly,

you begin asking yourself

> "Do I really need this?"

Eventually,

you remove everything that isn't essential.

Only the important items remain.

L1 Regularization works in exactly the same way.

Every weight carries a "cost."

Large weights become expensive.

As training continues,

the optimizer naturally starts asking

> "Do I really need this weight?"

If the answer is no,

it gradually pushes that weight toward zero.

---

# 📖 Chapter 9 — The Mathematics of L1

The penalty introduced by L1 Regularization is simply the sum of the absolute values of all model weights.

Mathematically,

```text
L1 Penalty = |w₁| + |w₂| + |w₃| + ... + |wn|
```

The complete objective function becomes

```text
Total Loss

=

Prediction Loss

+

λ ( |w₁| + |w₂| + ... + |wn| )
```

where

```text
λ (Lambda)
```

controls how strongly we penalize large weights.

---

# 🤔 Why Absolute Values?

A weight can be

```text
+10

or

-10
```

Both indicate a strong influence on the prediction.

If we simply summed the weights,

positive and negative values could cancel each other.

For example,

```text
+10

+

-10

=

0
```

Clearly,

this would not represent the true complexity of the model.

Absolute values solve this problem.

```text
|+10| = 10

|-10| = 10
```

Now every large weight contributes equally to the penalty,

regardless of its sign.

---

# 📖 Chapter 10 — The Role of λ (Lambda)

Lambda controls the balance between

```text
Prediction Accuracy

and

Model Simplicity
```

Imagine three different values.

### λ = 0

```text
No penalty.
```

The optimizer only minimizes prediction error.

This often leads to overfitting.

---

### Small λ

```text
Small penalty.
```

The optimizer still prioritizes prediction accuracy,

but slightly discourages unnecessarily large weights.

This is usually a good balance.

---

### Very Large λ

```text
Huge penalty.
```

The optimizer aggressively shrinks almost every weight.

The model becomes too simple,

leading to underfitting.

Note: λ does not directly shrink the weights. Instead, it controls how expensive large weights become in the optimization objective. During Gradient Descent, the optimizer balances two competing goals: reducing prediction error and reducing the penalty on large weights. A weight survives only if the improvement it brings to prediction is worth the extra penalty it incurs.
---

# 💡 Engineering Insight

Choosing λ is a trade-off.

```text
Too Small

↓

Overfitting

Too Large

↓

Underfitting
```

In practice,

engineers usually determine the best value of λ using validation data or cross-validation.

---

# 📖 Chapter 11 — The Most Beautiful Property of L1

Now comes the most fascinating part.

Imagine a model with the following feature weights.

```text
Area            = 3.8

Bedrooms        = 1.9

Location        = 4.2

Parking         = 0.7

Wall Color      = 0.03

Owner Name      = 0.01
```

Notice something.

Some features contribute almost nothing.

Because L1 continuously penalizes every weight,

these tiny weights eventually become

```text
0
```

The final model becomes

```text
Area            = 3.8

Bedrooms        = 1.9

Location        = 4.2

Parking         = 0.7

Wall Color      = 0

Owner Name      = 0
```

The useless features disappear completely.

This phenomenon is called

# Feature Selection.

---

# Chapter 12 — But Doesn't L1 Penalize Important Features Too?

At this point, you might have an important question.

> **"If L1 penalizes every weight, won't it also shrink the weights of important features like Area or Location?"**

At first glance, that seems true.

Suppose after training, our House Price model has learned the following weights.

```text
Area        = 3.8

Bedrooms    = 1.9

Location    = 4.2

Parking     = 0.7

Wall Color  = 0.03

Owner Name  = 0.01
```

L1 Regularization applies a penalty to **every** weight.

So why don't all of them eventually become zero?

The answer lies in something we've already learned in the **Optimization** module.

The optimizer is **not** minimizing only the Regularization Loss.

It is minimizing the **Total Loss**.

```text
Total Loss

=

Prediction Loss

+

λ × Regularization Loss
```

During every weight update, the optimizer computes gradients for **both** parts of the loss.

Mathematically,

```text
∂(Total Loss)/∂w

=

∂(Prediction Loss)/∂w

+

λ × ∂(Regularization Loss)/∂w
```

and Gradient Descent updates the weights as

```text
w_new

=

w_old

-

η ×

(

∂(Prediction Loss)/∂w

+

λ × ∂(Regularization Loss)/∂w

)
```

You don't need to memorize this equation.

Instead, focus on what it means.

There are now **two competing forces** acting on every weight.

### Force 1 — Improve Prediction Accuracy

This part of the gradient asks:

> "How should I change this weight to reduce the prediction error?"

If a feature is important for making accurate predictions, this force strongly tries to preserve its weight.

---

### Force 2 — Keep the Model Simple

The regularization term asks:

> "Can this weight be made smaller without hurting prediction accuracy too much?"

This force tries to shrink every weight toward zero.

---

The optimizer balances both objectives simultaneously.

Think of it as two managers discussing every weight update.

```text
Manager 1

↓

"I care about prediction accuracy."

Keep important weights large.


Manager 2

↓

"I care about keeping the model simple."

Reduce unnecessary weights.
```

The optimizer finds a compromise between them.

Let's return to our example.

```text
Area        = 3.8

Bedrooms    = 1.9

Location    = 4.2

Parking     = 0.7

Wall Color  = 0.03

Owner Name  = 0.01
```

Suppose reducing the **Area** weight causes the prediction error to increase significantly.

The Prediction Loss immediately pushes that weight back up.

As a result, the optimizer keeps it relatively large.

Now consider **Owner Name**.

Reducing this weight barely changes the prediction error because the feature contributes almost nothing.

The Prediction Loss has little reason to keep it.

The Regularization Loss gradually pushes it toward zero.

Eventually,

```text
Owner Name

↓

0
```

This is exactly how **L1 Regularization performs automatic feature selection**.

It doesn't remove features simply because they have large weights.

It removes features whose contribution to prediction is too small to justify keeping them.

L1 tends to produce sparse solutions. When multiple correlated features provide similar predictive power, it often assigns most or all of the weight to one feature while shrinking the others toward zero. Which feature survives depends on the optimization path, initialization, feature scaling, and small differences in the data.
---

## 💡 Engineering Insight

One common misconception is:

> "L1 shrinks every weight equally."

That's only half the story.

L1 **tries** to shrink every weight.

But the Prediction Loss fights to preserve weights that are genuinely useful.

Only the weights that contribute very little to prediction are eventually driven all the way to zero.

This balance between **prediction accuracy** and **model simplicity** is what makes L1 Regularization so effective.
---

# 📖 Chapter 13 — Why Feature Selection Matters

Imagine building a disease prediction model using

```text
500 medical tests.
```

Do doctors really need all 500?

Probably not.

L1 Regularization automatically removes tests that contribute little to the prediction.

The resulting model becomes

- simpler,
- faster,
- easier to interpret,
- and often more robust.

This is one reason L1 is widely used in domains where interpretability matters.

Examples include

- Medical Diagnosis
- Credit Risk Assessment
- Fraud Detection
- Genomics
- Text Classification

---

# 📖 Chapter 14 — Does L1 Always Win?

Although L1 is powerful,

it isn't perfect.

Suppose two features contain almost identical information.

```text
House Area

Built-up Area
```

Both are highly correlated.

L1 often keeps one feature

and removes the other entirely.

Sometimes,

both features are actually useful.

In such cases,

L1 may remove more information than we would like.

Researchers therefore began asking another question.

> "Can we reduce large weights without forcing them all the way to zero?"

That question eventually led to another regularization technique...

**L2 Regularization.**

---

# 🎯 Key Takeaways

- Regularization discourages overly complex models.
- L1 penalizes the absolute values of the weights.
- Lambda (λ) controls the strength of the penalty.
- Small λ → Higher risk of overfitting.
- Large λ → Higher risk of underfitting.
- L1 naturally pushes many weights exactly to zero.
- This makes L1 an automatic feature selection technique.
- L1 is especially useful when model interpretability is important.

---

# 🤔 Curiosity Corner

L1 removes unnecessary features by pushing many weights to exactly zero.

But what if we don't actually want to eliminate features?

What if we simply want to make **all** weights smaller,

while keeping every feature in the model?

Can we design a penalty that shrinks weights...

without deleting them?

That idea gave birth to one of the most widely used regularization techniques in Machine Learning...

**L2 Regularization.**


Q: "If the regularization loss is just the sum of the weights, aren't the weights constants? Shouldn't its derivative be zero?"

A: No. During optimization, the weights are the variables, not constants. The optimizer treats the regularization loss as a function of the weights and computes how that loss changes if each weight changes slightly. That's exactly why the regularization term can influence every weight update.