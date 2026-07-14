# 🎯 Lecture 1 - Perceptron Interview Questions

> **Objective:** Prepare for Deep Learning and GenAI interviews from beginner to Staff Engineer level.

---

# 🟢 Level 1 — Beginner

## Q1. What is a Perceptron?

### Expected Answer

A Perceptron is the simplest neural network that computes a weighted sum of inputs, adds a bias, applies an activation function, and produces a binary prediction.

---

## Q2. What are the main components of a Perceptron?

Expected points

- Input Features
- Weights
- Bias
- Weighted Sum
- Activation Function
- Output

---

## Q3. What is the role of weights?

Expected Answer

Weights determine:

- Direction of influence (positive/negative)
- Magnitude of influence (strength)

---

## Q4. Why do we need Bias?

Expected Answer

Bias shifts the decision boundary away from the origin, allowing the model to learn a wider range of functions.

Without bias, every decision boundary must pass through the origin.

---

## Q5. What activation function does a Perceptron use?

Answer

Step Function

---

# 🔵 Level 2 — Intermediate

## Q6. Why is a Perceptron still a linear model?

Expected Answer

Although the Step Function looks non-linear, the weighted sum

```text
Wx + b
```

creates only one linear decision boundary.

The activation only thresholds the output.

Therefore the expressive power remains linear.

---

## Q7. Can a Perceptron solve XOR?

Answer

No.

Reason:

XOR is not linearly separable.

A single hyperplane cannot separate XOR classes.

---

## Q8. Why can't Linear Regression recognize faces?

Expected Answer

Because Linear Regression learns only linear relationships.

Face recognition requires extracting complex **hierarchical features** from raw pixels, which linear models cannot learn automatically.

---

## Q9. Why is Feature Engineering difficult in NLP?

Expected Answer

Language is highly contextual.

The same word can have different meanings depending on context.

It is impossible to manually engineer rules for every linguistic scenario.

---

## Q10. What is Representation Learning?

Expected Answer

Representation Learning is the process by which neural networks automatically learn useful features directly from raw data instead of relying on manually engineered features.

---

# 🟠 Level 3 — Senior Engineer

## Q11. Why is Representation Learning considered revolutionary?

Expected Discussion

Traditional ML

```text
Raw Data
     ↓
Human Feature Engineering
     ↓
Model
```

Deep Learning

```text
Raw Data
     ↓
Neural Network
     ↓
Learns Features Automatically
```

Removing manual feature engineering enabled breakthroughs in vision, speech and NLP.

---

## Q12. Why can't stacking multiple linear layers solve non-linearity?

Expected Answer

**Composition of linear functions is still linear.**

Example

```text
Layer 1

h₁ = W₁x + b₁

Layer 2

h₂ = W₂h₁ + b₂
```

Substituting

```text
h₂ = W₂(W₁x+b₁)+b₂
```

Expanding

```text
h₂ = (W₂W₁)x + (W₂b₁+b₂)
```

Still

```text
Wx+b
```

Hence multiple linear layers remain a linear transformation.

---

## Q13. What happens if we remove Bias from every neuron?

Expected Answer

Every decision boundary would pass through the origin.

The expressive power of the network would decrease.

Training would become harder because neurons lose flexibility.

---

## Q14. Why is the Step Function not trainable?

Expected Answer

Because it is not differentiable at the threshold and has zero derivative almost everywhere else.

Gradient Descent requires gradients to update weights.

Therefore learning cannot occur.

---

# 🔴 Level 4 — Staff Engineer

## Q15. GPT contains billions of neurons.

Is every neuron basically a Perceptron?

Expected Answer

Conceptually yes.

Every neuron computes

```text
Weighted Sum

↓

Bias

↓

Activation
```

However GPT replaces the Step Function with differentiable activations like GELU.

---

## Q16. Suppose GPT used Step Function instead of GELU.

What would happen?

Expected Discussion

- No meaningful gradients
- Backpropagation fails
- No learning
- Training becomes impossible

---

## Q17. If Perceptron introduced weights and bias, why wasn't Deep Learning successful in the 1960s?

Expected Discussion

Points to mention

- No hidden layers
- No differentiable activation
- XOR limitation
- Limited compute
- Limited datasets
- No GPU
- No Backpropagation

---

## Q18. Can Decision Trees replace LLMs?

Expected Answer

No.

Decision Trees require manually engineered features and build rule-based splits.

LLMs automatically learn hierarchical representations from raw data and generalize much better for language understanding.

---

# ⭐ Rapid Fire (30 Second Answers)

- What is a Perceptron?
- Why do we need weights?
- Why do we need bias?
- Why is Perceptron linear?
- Can it solve XOR?
- Why not?
- What is Representation Learning?
- Why is Feature Engineering difficult?
- Difference between Linear Regression and Perceptron?
- Why isn't Step Function used today?
- What replaced Step Function?
- Why do modern neural networks need hidden layers?

---

# 🧠 Follow-up Question Tree

```text
Perceptron
     │
     ├── Why weights?
     │      └── Direction + Magnitude
     │
     ├── Why bias?
     │      └── Shift decision boundary
     │
     ├── Why activation?
     │      └── Binary decision
     │
     ├── Why still linear?
     │      └── Single hyperplane
     │
     ├── Why can't solve XOR?
     │      └── Non-linear data
     │
     ├── How do we solve XOR?
     │      └── Hidden layers
     │
     └── Why hidden layers?
            └── Learn non-linear representations
```

---

# 💡 Common Interview Mistakes

❌ Saying

> Weight means importance.

✅ Better

> Weight determines both the direction and magnitude of a feature's influence.

---

❌ Saying

> Bias shifts the graph.

✅ Better

> Bias allows the neuron to activate even when weighted inputs alone are insufficient, increasing the flexibility of the learned decision boundary.

---

❌ Saying

> Step Function makes Perceptron non-linear.

✅ Better

> Although the Step Function is non-linear, the model still produces a single linear decision boundary, so the overall classifier remains linear.

---

# 🚀 What the Interviewer is Really Testing

When interviewers ask about Perceptron, they are usually checking whether you understand:

- Linear vs Non-linear models
- Representation Learning
- Role of weights and bias
- Why activation functions exist
- Why hidden layers became necessary
- Historical evolution toward Deep Learning
- Whether you can connect classical neural networks to modern architectures like GPT

If you can explain these concepts clearly with reasoning—not just definitions—you'll stand out from candidates who have only memorized textbook answers.
