# Training Neural Networks — Interview Discussion

---

# About This Document

This document is **not** a collection of memorization-based interview questions.

Instead, it contains the kinds of discussions that typically happen during Senior, Staff, and Lead AI Engineer interviews.

Many questions intentionally build upon previous ones, simulating how interviewers gradually probe your depth of understanding.

---

# Section 1 — Loss Functions

## Q1. Why can't we use Accuracy as the loss function?

**Expected Discussion**

- Accuracy is not differentiable.
- It only tells whether a prediction is correct or incorrect.
- It provides no information about *how wrong* a prediction is.
- Gradient Descent requires smooth gradients to update weights.

---

## Q2. What properties should a good loss function have?

Expected points:

- Non-negative
- Small mistakes → small penalty
- Large mistakes → large penalty
- Smooth and differentiable
- Guides learning through gradients

---

## Q3. Why should a loss function be smooth?

Follow-up:

How does a non-smooth loss affect Backpropagation?

---

## Q4. Why does the loss guide learning?

Can you explain this without using equations?

---

## Q5. Suppose your loss suddenly becomes zero.

Does this always mean your model is perfect?

---

# Section 2 — Regression Losses

## Q6. When would you choose MSE over MAE?

Discuss:

- Outliers
- Robustness
- Safety-critical applications
- Optimization behavior

---

## Q7. Why does MSE square the error?

Follow-up:

What would happen if we simply averaged the raw errors?

---

## Q8. Why does MSE naturally take large optimization steps initially and smaller steps near convergence?

---

## Q9. Which model would MSE prefer?

Model A

```text
Errors

1
2
1
2
```

Model B

```text
Errors

0
0
0
6
```

Explain your reasoning.

---

# Section 3 — Classification Losses

## Q10. Why isn't MSE commonly used for classification?

---

## Q11. Why is Cross Entropy preferred?

---

## Q12. Why does Cross Entropy use the logarithm?

Can you explain the intuition without deriving the equation?

---

## Q13. What happens if the model predicts the correct class with probability 1?

What should the Cross Entropy loss become?

---

## Q14. Why does predicting the wrong class with very high confidence receive such a large penalty?

---

## Q15. Explain Binary Cross Entropy.

Where is it used?

---

## Q16. What problem does BCEWithLogitsLoss solve?

Why is it preferred over applying Sigmoid followed by Binary Cross Entropy manually?

---

# Section 4 — Gradient Descent

## Q17. What exactly is a gradient?

How is it different from a derivative?

---

## Q18. Why do we move in the direction of the negative gradient?

---

## Q19. What happens if the learning rate is too small?

---

## Q20. What happens if the learning rate is too large?

---

## Q21. Why do we need partial derivatives?

---

## Q22. What is the Gradient Vector?

Why do modern neural networks require it?

---

# Section 5 — Backpropagation

## Q23. Explain Backpropagation to someone who has never studied calculus.

---

## Q24. Why do we need the Chain Rule?

---

## Q25. Why does the Chain Rule multiply derivatives instead of adding them?

---

## Q26. During Backpropagation, what exactly is flowing backward?

Is it the loss?

The weights?

The gradients?

Explain carefully.

---

## Q27. Why don't we update the weights during the Forward Pass?

---

# Section 6 — Computational Graph

## Q28. What is a Computational Graph?

Why do frameworks like PyTorch build it automatically?

---

## Q29. What information is stored inside a Computational Graph?

---

## Q30. Why is the Computational Graph discarded after each iteration?

---

# Section 7 — Training Pipeline

## Q31. Explain the complete training pipeline of a neural network.

Starting from the input,

walk through every step until the weights are updated.

---

## Q32. What happens during one Forward Pass?

---

## Q33. What happens during one Backward Pass?

---

## Q34. What is an Epoch?

How is it different from an Iteration?

---

## Q35. Suppose the loss decreases every epoch.

Does this always imply that the model is improving?

---

# Section 8 — Scenario-Based Questions

## Q36.

Suppose your training loss decreases,

but your validation loss starts increasing.

What does this indicate?

---

## Q37.

Suppose the loss never decreases.

List every possible reason you would investigate.

---

## Q38.

Suppose every gradient becomes zero after a few iterations.

What are the possible causes?

---

## Q39.

Suppose your model predicts exactly the same output for every input.

How would you debug it?

---

## Q40.

Suppose training suddenly becomes unstable after reducing the batch size.

What could be happening?

---

# Section 9 — Deep Thinking Questions

## Q41.

Imagine you're inventing Deep Learning from scratch.

How would you design a loss function?

What properties would it satisfy?

---

## Q42.

If derivatives had never been invented,

could neural networks still learn?

How?

---

## Q43.

Suppose Backpropagation didn't exist.

Can you think of another way to optimize billions of parameters?

---

## Q44.

Why is training a neural network considered an optimization problem rather than a search problem?

---

## Q45.

Imagine you're explaining neural network training to a school student.

How would you describe the complete learning process without using any mathematical terms?

---

# Final Interview Challenge

Without looking at any notes,

explain the complete learning process of a neural network in under **5 minutes**.

Your explanation should naturally include:

- Forward Pass
- Predictions
- Loss Function
- Computational Graph
- Backpropagation
- Chain Rule
- Gradients
- Gradient Descent
- Weight Updates
- Epochs

If you can explain this clearly and confidently, you've understood the foundation of how every modern deep learning model—including GPT—is trained.


---

# 🎯 Staff Engineer Challenge

The following questions are intentionally open-ended.

Unlike the previous interview questions, these do **not** have a single correct answer.

The goal is to evaluate your ability to reason about system design, trade-offs, and engineering decisions.

These are the kinds of discussions commonly seen in **Staff Engineer, Principal Engineer, and AI Architect** interviews.

---

## Challenge 1 — Design Your Own Loss Function

Imagine you are inventing deep learning from scratch.

You need to design a new loss function for a regression problem.

### Discussion Points

- What properties should your loss function satisfy?
- How should it treat small errors?
- How should it treat large errors?
- Should it be differentiable?
- How would you verify that Gradient Descent can optimize it?

---

## Challenge 2 — Training Loss Never Decreases

Suppose your team has been training a neural network for several hours.

The training loss remains almost constant.

As the technical lead, how would you investigate the problem?

### Discussion Points

Consider every possible cause.

Examples include:

- Learning rate
- Incorrect loss function
- Wrong labels
- Gradient computation
- Weight initialization
- Data preprocessing
- Activation functions
- Bugs in the training pipeline

How would you systematically debug the issue?

---

## Challenge 3 — Build a Learning Pipeline from Scratch

Without using TensorFlow or PyTorch,

design the complete training pipeline for a neural network.

Your explanation should include:

- Forward Pass
- Loss Computation
- Computational Graph
- Backpropagation
- Gradient Computation
- Weight Updates
- Training Loop

---

## Challenge 4 — Explain Neural Network Training to a CEO

Your CEO asks,

> "Why does training GPT take weeks on thousands of GPUs? Why can't we simply tell the model the correct answers?"

How would you explain the entire learning process in simple, non-technical language?

---

## Challenge 5 — When Would You NOT Use Mean Squared Error?

Suppose you are working on different AI applications.

Would you choose MSE for each of the following?

- House Price Prediction
- Autonomous Driving
- Credit Card Fraud Detection
- Medical Diagnosis
- Machine Translation
- Next-Word Prediction in GPT

If not,

what alternative loss function would you choose, and why?

---

## Challenge 6 — Backpropagation Does Not Exist

Imagine Backpropagation had never been invented.

How else could a neural network learn?

Think creatively.

Possible directions to explore include:

- Evolutionary Algorithms
- Random Search
- Reinforcement Learning
- Finite Difference Methods
- Other optimization strategies

Discuss the advantages and limitations of each approach.

---

## Challenge 7 — Designing a Modern Deep Learning Framework

Suppose you are one of the engineers building the next version of PyTorch.

Which features would you implement to make training neural networks faster, more stable, and easier to debug?

Consider topics such as:

- Automatic Differentiation
- Computational Graphs
- GPU Acceleration
- Mixed Precision Training
- Gradient Checkpointing
- Distributed Training
- Model Debugging Tools

---

## Challenge 8 — The Ultimate First-Principles Test

Without writing a single equation,

explain how a neural network learns.

Your explanation should naturally cover:

- Predictions
- Errors
- Loss Functions
- Gradients
- Backpropagation
- Weight Updates
- Learning through repeated experience

If a school student can understand your explanation,

you truly understand the concept yourself.

---

# Final Thought

A Staff Engineer is not expected to memorize formulas.

A Staff Engineer is expected to reason about systems.

Whenever you attempt these challenges, focus on **why** a design choice is made, what trade-offs it introduces, and how you would justify your decision to another experienced engineer.