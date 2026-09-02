# 📝 Assignment 1 — Perceptron

> **Objective:** Verify that you understand the Perceptron from first principles rather than by memorizing definitions.

---

# 🎯 Instructions

- Write answers in your own words.
- Avoid copying textbook definitions.
- Wherever possible, explain **why** instead of **what**.
- If you are unsure, write your reasoning. We'll improve it together.

---

# Part A — Conceptual Questions

## Q1. Why can't Linear Regression recognize faces?

Expected discussion

- Linear relationship
- Raw pixels are not meaningful features
- Manual feature engineering
- Complex non-linear relationships

---

## Q2. Why is feature engineering difficult for Natural Language Processing?

Discuss

- Context
- Ambiguity
- Same word, multiple meanings
- Impossible to manually define every rule

---

## Q3. Explain Representation Learning in your own words.

Do **not** use the textbook definition.

Explain it as if you were teaching a junior engineer.

---

## Q4. Why do deeper networks learn more complex features than shallow networks?

Hint

Think about feature hierarchies.

---

## Q5. Can a Decision Tree replace an LLM?

Explain your reasoning.

---

# Part B — Neural Network Fundamentals

## Q6. What is the role of weights?

Don't answer "importance."

Explain

- Direction
- Magnitude
- Influence

---

## Q7. Why do we need bias?

Don't simply say

> "It shifts the graph."

Explain

- What happens without bias?
- What kinds of functions become impossible to learn?

---

## Q8. Why is the weighted sum still a linear operation?

Hint

Look carefully at

```text
z = Wx + b
```

---

## Q9. Is a single Perceptron more powerful than Linear Regression?

Support your answer mathematically.

---

## Q10. Why is the Step Function used?

What problem was it trying to solve?

---

# Part C — Think Like an Engineer

## Q11.

Suppose you are building the first neural network in history.

Would you choose

- No activation
- Step Function
- ReLU

Explain your reasoning as if you were living in the 1950s and ReLU had not yet been invented.

---

## Q12.

Suppose someone tells you

> "Let's build GPT using only one Perceptron."

Explain why that cannot work.

---

## Q13.

Imagine we remove the bias term from every neuron inside GPT.

What would you expect to happen during training?

---

## Q14.

If the Step Function is non-linear,

why does the Perceptron still remain a linear classifier?

---

# ⭐ Part D — Staff Engineer Challenge

Imagine you are explaining the Perceptron to a new graduate joining your team.

You have exactly **3 minutes**.

Your explanation should include:

- Why Perceptron was invented
- What problem it solved
- Why it ultimately failed
- Why modern Deep Learning still builds upon it

Maximum length

300 words.

---

# 🎤 Mock Interview Exercise

Record yourself answering these questions without looking at your notes.

1. What is a Perceptron?
2. Why do we need weights?
3. Why do we need bias?
4. Why is Perceptron linear?
5. Can it solve XOR?
6. Why not?
7. What is Representation Learning?
8. Why can't Linear Regression recognize faces?

Target

- 30–60 seconds per answer
- Clear explanation
- No memorization
- Use examples whenever possible

---

# 📚 Recommended Reading

Before moving to the next lecture, make sure you can confidently explain:

- Perceptron Architecture
- Weights
- Bias
- Weighted Sum
- Activation Function
- Representation Learning
- Linear Decision Boundary
- XOR Limitation

---

# ✅ Completion Checklist

Mark each item after you complete it.

- [ ] Read the lecture
- [ ] Read the revision sheet
- [ ] Completed all conceptual questions
- [ ] Completed engineer-level questions
- [ ] Completed Staff challenge
- [ ] Practiced mock interview aloud
- [ ] Reviewed weak areas
- [ ] Ready for Lecture 2

---

# 🚀 Success Criteria

You are ready to move to the next lecture if you can answer:

- Why is a Perceptron linear?
- Why can't it solve XOR?
- Why do we need weights?
- Why do we need bias?
- Why was Representation Learning revolutionary?
- How is a Perceptron related to GPT?

without referring to your notes.
