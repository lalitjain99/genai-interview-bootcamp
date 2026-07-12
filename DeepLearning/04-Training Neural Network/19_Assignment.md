# Training Neural Networks — Assignment

---

# About This Assignment

These assignments are designed to strengthen your understanding of how neural networks learn.

The objective is **not** to write code or memorize equations.

Instead, each question encourages you to think like an ML Engineer who is designing deep learning systems from first principles.

---

# 🟢 Level 1 — Conceptual Understanding

## Assignment 1 — Explain Neural Network Learning

Without using any mathematical equations,

explain how a neural network learns.

Your explanation should naturally include:

- Prediction
- Error
- Loss Function
- Gradients
- Backpropagation
- Weight Updates

**Constraint**

Imagine you are explaining this to someone who has never studied machine learning.

---

## Assignment 2 — Design the Perfect Loss Function

Imagine you are inventing deep learning from scratch.

Design a loss function for a regression problem.

Your loss function should satisfy the following properties:

- Never becomes negative.
- Small mistakes receive small penalties.
- Large mistakes receive larger penalties.
- Can be optimized using Gradient Descent.

Explain why each property is necessary.

---

## Assignment 3 — Compare Regression Loss Functions

Compare the following loss functions.

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

Discuss:

- Advantages
- Limitations
- Sensitivity to outliers
- Optimization behavior
- Real-world applications

Finally,

recommend one application where MSE is preferred,

and another where MAE would be a better choice.

---

# 🟡 Level 2 — Engineering Thinking

## Assignment 4 — Debug a Training Failure

Suppose your neural network has been training for several hours.

However,

the loss never decreases.

Prepare a debugging checklist.

Your checklist should include every possible reason you would investigate,

including topics such as:

- Data
- Labels
- Learning Rate
- Gradients
- Weight Initialization
- Activation Functions
- Training Loop
- Loss Function

---

## Assignment 5 — Build the Training Pipeline

Draw the complete training pipeline of a neural network.

Your diagram should include:

```text
Input

↓

Forward Pass

↓

Prediction

↓

Loss Function

↓

Computational Graph

↓

Backpropagation

↓

Gradient Computation

↓

Gradient Descent

↓

Updated Weights
```

For every stage,

write **one sentence** explaining its purpose.

---

## Assignment 6 — Explain Backpropagation Without Mathematics

Suppose your teammate has never studied calculus.

Explain Backpropagation without using any equations.

Use an analogy from everyday life.

The goal is to explain **why information must flow backward** through the network.

---

# 🔴 Level 3 — First-Principles Thinking

## Assignment 7 — Invent a Bad Loss Function

Design a loss function that would completely fail to train a neural network.

Examples of bad ideas include:

- Always returning a constant value.
- Returning random numbers.
- Producing discontinuous outputs.

For each design,

explain **why Gradient Descent would fail.**

---

## Assignment 8 — If Backpropagation Never Existed

Imagine Backpropagation had never been invented.

Suggest alternative ways to train a neural network.

Examples might include:

- Evolutionary Algorithms
- Random Search
- Reinforcement Learning
- Finite Difference Methods

Discuss the advantages and limitations of each approach.

---

## Assignment 9 — Explain Why GPT Can Learn

GPT is trained using billions of sentences,

yet no human manually labels every next word.

Explain,

step by step,

how GPT creates its own training examples.

Your explanation should naturally include:

- Tokens
- Input Sequence
- Ground Truth
- Next-Token Prediction
- Cross Entropy
- Gradient Descent
- Weight Updates

---

# 🎯 Capstone Challenge

Imagine you have been asked to build a deep learning framework from scratch.

Describe the complete training process.

Your explanation should include:

- Forward Pass
- Loss Computation
- Automatic Differentiation
- Computational Graph
- Backpropagation
- Gradient Computation
- Weight Updates
- Epochs
- Mini-Batches

Finally,

explain why each component is necessary.

If any one component were removed,

what would happen to training?

---

# Bonus Challenge

Throughout this module,

we explored several "What If?" scenarios.

Answer the following without referring to your notes.

1. What happens if we add a constant to the loss?
2. What happens if we multiply the loss by a constant?
3. Why does MSE naturally guide learning?
4. Why does Cross Entropy use the logarithm?
5. Why can't we use accuracy as a loss function?
6. Why do gradients naturally become smaller as training converges?

Explain your reasoning using intuition rather than memorized formulas.