# 📌 Cheat Sheet — Sigmoid Activation Function

---

# 🎯 Why Was Sigmoid Invented?

The Step Function could make decisions but **could not learn** because it was not differentiable.

Researchers wanted an activation function that:

- ✅ Introduces non-linearity
- ✅ Is differentiable
- ✅ Allows gradients to flow
- ✅ Enables Gradient Descent
- ✅ Produces meaningful outputs

Sigmoid solved this problem.

---

# 🧮 Sigmoid Formula

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

---

# 📈 Output Range

```text
Input → Output

z = +∞  → 1

z = 0   → 0.5

z = -∞  → 0
```

Output always lies between:

```text
0 and 1
```

---

# 🎯 Why Is Sigmoid Useful?

- Smooth approximation of the Step Function
- Differentiable everywhere
- Supports Backpropagation
- Output can be interpreted as confidence
- Commonly used for Binary Classification output layers

---

# 🔄 Learning Pipeline

```text
Input
   ↓
Weighted Sum (Wx+b)
   ↓
Sigmoid
   ↓
Prediction
   ↓
Loss
   ↓
Gradient
   ↓
Backpropagation
   ↓
Weight Update
```

---

# 📉 Vanishing Gradient Problem

Sigmoid derivative is always:

```text
0 < σ'(z) ≤ 0.25
```

During Backpropagation:

```text
Small Gradient

×

Small Gradient

×

Small Gradient

↓

Almost Zero
```

Result:

- Early layers stop learning
- Training becomes slow
- Deep networks become difficult to optimize

---

# ⚖️ Step Function vs Sigmoid

| Step Function | Sigmoid |
|---------------|---------|
| Hard decision | Soft decision |
| Not differentiable | Differentiable |
| No Gradient Descent | Supports Gradient Descent |
| Output: 0 or 1 | Output: 0 to 1 |
| Cannot train deep networks | Can train neural networks |

---

# 💡 Where Is Sigmoid Used Today?

✅ Binary Classification Output Layer

✅ Logistic Regression

✅ LSTM / GRU Gates

---

# 🚫 Where Is Sigmoid NOT Used?

❌ Hidden layers of deep neural networks

❌ CNN hidden layers

❌ Transformers

❌ Large Language Models

---

# 🧠 Interview One-Liners

**Why did Sigmoid replace the Step Function?**

> Because it is differentiable, allowing Gradient Descent and Backpropagation.

---

**Why isn't Sigmoid used in modern hidden layers?**

> Because repeated multiplication of its small gradients causes the Vanishing Gradient Problem.

---

**Why is Sigmoid suitable for binary classification?**

> Its output lies between 0 and 1 and can be interpreted as confidence for the positive class.

---

# 🎯 Key Takeaways

- Step Function could decide but couldn't learn.
- Sigmoid is a smooth version of the Step Function.
- Sigmoid enabled Backpropagation.
- Sigmoid outputs values between 0 and 1.
- Deep networks expose Sigmoid's Vanishing Gradient Problem.
- Modern deep networks prefer ReLU or GELU for hidden layers.