# Activation Functions - Assignment

---

# 🎯 Objective

This assignment is designed to test your conceptual understanding rather than your ability to memorize formulas.

Try answering every question in your own words before looking anything up.

---

# 🟢 Level 1 – Foundation

## Q1. Why do neural networks need activation functions?

Don't answer "to introduce non-linearity."

Explain what happens mathematically if every activation function is removed from a deep neural network.

---

## Q2. Why can't stacking multiple linear layers solve non-linear problems?

Hint:

Consider two linear layers.

```text
Layer 1

h = W₁x + b₁

Layer 2

y = W₂h + b₂
```

Substitute Layer 1 into Layer 2.

What do you observe?

---

## Q3. Why was the Step Function suitable for the Perceptron but unsuitable for modern Deep Learning?

Explain both:

- Forward pass
- Backpropagation

---

## Q4. Explain the role of an activation function using a real-world analogy.

Do not use any mathematical equations.

---

# 🟡 Level 2 – Interview Level

## Q5. Why does Sigmoid suffer from the Vanishing Gradient Problem?

Your answer should include:

- Sigmoid derivative
- Chain Rule
- Deep networks
- Gradient Descent

---

## Q6. ReLU is linear for positive inputs.

Then why is ReLU considered a non-linear activation function?

This is one of the most common interview questions.

---

## Q7. Explain Sparse Activation.

Why does it improve computational efficiency?

---

## Q8. Explain the Dying ReLU problem mathematically.

Use the Gradient Descent update equation.

```text
w(new) = w(old) − η × ∂L/∂w
```

Expand the gradient using the Chain Rule.

---

## Q9. Why can poor weight initialization create dead neurons?

Imagine a neuron whose weighted sum is always negative.

Explain why it struggles to recover.

---

## 🟠 Level 3 – Senior Engineer

## Q10. Why was Leaky ReLU introduced?

Instead of saying

> "It fixes Dying ReLU"

explain the mathematical reason.

Why does introducing a small negative slope help Gradient Descent?

---

## Q11. Why doesn't Leaky ReLU completely solve the limitations of ReLU?

Think beyond dead neurons.

Consider the shape of the function itself.

---

## Q12. Explain the philosophical difference between ReLU and GELU.

Do not discuss equations.

Instead explain how each activation function "thinks."

Hint:

One makes binary decisions.

The other makes probabilistic decisions.

---

## Q13. Why is language understanding fundamentally different from image classification with respect to activation functions?

Why might hard thresholding work reasonably well for vision but not be ideal for language?

---

# 🔴 Level 4 – Staff Engineer

## Q14. Why does GELU multiply the input by a probability?

Hint:

Suppose the neuron output is

```text
10
```

Would returning only the probability preserve the original information?

Why or why not?

---

## Q15. Why does GELU use the Gaussian CDF instead of the Gaussian PDF?

Explain the intuition.

No mathematical proof required.

---

## Q16. GPT already has Softmax inside the Attention layer.

Why does it still need GELU inside the Feed Forward Network?

This question checks whether you understand the different roles of Softmax and GELU.

---

## Q17. Suppose we replace every GELU inside GPT with ReLU.

Predict the consequences.

Consider:

- Optimization
- Gradient Flow
- Language Understanding
- Representation Learning

---

## Q18. Suppose we remove activation functions completely from GPT.

Would Attention alone be sufficient to learn complex language patterns?

Justify your answer carefully.

---

# ⭐ Bonus Challenge

Imagine you are designing a brand-new activation function for the next generation of LLMs.

Answer the following:

1. What limitations of GELU would you try to solve?
2. What properties should your activation function have?
3. Would it use probability?
4. Would it be computationally cheaper than GELU?
5. How would you convince researchers to adopt it?

There is no single correct answer.

This question is designed to evaluate your depth of understanding and creativity.

---

# 📚 Self-Evaluation Checklist

After completing this assignment, you should be able to explain confidently:

- ✅ Why activation functions exist
- ✅ Why linear layers are insufficient
- ✅ Why Step Function failed
- ✅ Why Sigmoid replaced Step Function
- ✅ Why Sigmoid causes Vanishing Gradients
- ✅ Why ReLU became the default activation
- ✅ Why ReLU suffers from dead neurons
- ✅ Why Leaky ReLU was introduced
- ✅ Why GELU performs probabilistic gating
- ✅ Why GPT uses GELU
- ✅ Why Softmax and GELU serve different purposes

---

# 💡 Advice

Do not write one-line answers.

Pretend you are explaining these concepts to a junior engineer or answering them in a whiteboard interview.

If you can teach the concept clearly, you've truly understood it.