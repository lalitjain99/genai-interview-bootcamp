# ⚡ Optimizer Evolution — Cheat Sheet

> **2-Minute Revision: From Gradient Descent to AdamW**

---

# 🎯 One-Line Evolution

```text
Gradient Descent
      ↓
Momentum
      ↓
AdaGrad
      ↓
RMSProp
      ↓
Adam
      ↓
AdamW
```

> Every optimizer exists because the previous one had a limitation.

---

# 🚀 Evolution at a Glance

| Optimizer | Main Idea | Solved | New Limitation |
|------------|-----------|--------|----------------|
| 🟢 Gradient Descent | Learn using gradients | Basic optimization | No memory |
| 🟡 Momentum | Remember previous direction | Faster convergence | Same learning rate |
| 🔵 AdaGrad | Adaptive learning rate | Sparse features | Learning rate decays forever |
| 🟣 RMSProp | Forget old gradients | Stable learning rate | No momentum |
| 🔴 Adam | Momentum + RMSProp | Direction + Step Size | Large weights |
| 🟠 AdamW | Decoupled Weight Decay | Better generalization | Still needs tuning |

---

# 🧠 What Does Each Optimizer Remember?

| Optimizer | Memory |
|------------|--------|
| Gradient Descent | ❌ Nothing |
| Momentum | ✅ Average Gradient (EMA) |
| AdaGrad | ✅ All Historical Squared Gradients |
| RMSProp | ✅ Recent Squared Gradients (EMA) |
| Adam | ✅ Average Gradient + Average Squared Gradient |
| AdamW | ✅ Adam + Independent Weight Control |

---

# 🎯 Core Equations

### 🟢 Gradient Descent

```text
w = w − ηg
```

---

### 🟡 Momentum

```text
v = βv + (1−β)g
```

---

### 🔵 AdaGrad

```text
w = w − ηg / √G
```

---

### 🟣 RMSProp

```text
v = βv + (1−β)g²
```

---

### 🔴 Adam

```text
m̂
↓

v̂

↓

w = w − η m̂ / √v̂
```

---

### 🟠 AdamW

```text
Adam Update

+

Weight Decay
```

---

# 🚗 Car Analogy

| Driving | Optimizer |
|----------|-----------|
| 🚗 Start Moving | Gradient Descent |
| 🧭 Keep Steering Stable | Momentum |
| ⚡ Adjust Speed per Wheel | AdaGrad |
| 🕒 Ignore Old Road Conditions | RMSProp |
| 🚘 Steering + Speed Together | Adam |
| 🛑 Prevent Overspeeding | AdamW |

---

# 📊 Feature Comparison

| Feature | GD | Mom | Ada | RMS | Adam | AdamW |
|----------|:--:|:---:|:---:|:---:|:----:|:-----:|
| Direction Memory | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Adaptive LR | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| EMA | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Bias Correction | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Weight Decay | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

# 🎯 Which Optimizer Should I Choose?

| Scenario | Optimizer |
|-----------|-----------|
| 📚 Small Dataset | SGD + Momentum |
| 🌵 Sparse Features | AdaGrad |
| 🧪 Research Prototype | Adam |
| 🤖 Deep Neural Networks | Adam |
| 🧠 Transformers | AdamW |
| 💬 Large Language Models | AdamW |

---

# 🌍 Industry Usage

| Domain | Preferred Optimizer |
|----------|--------------------|
| CNN | SGD + Momentum |
| Recommendation System | AdaGrad |
| RNN / LSTM | Adam |
| Transformer | AdamW |
| GPT | AdamW |
| LLaMA | AdamW |
| Diffusion Models | AdamW |

---

# 🎤 Interview Rapid Fire

- Why wasn't Gradient Descent enough?
- Why was Momentum invented?
- Why did AdaGrad fail?
- Why did RMSProp replace AdaGrad?
- Why combine Momentum and RMSProp?
- Why isn't Adam the final optimizer?
- Why is AdamW better than Adam?
- Why do LLMs use AdamW?

---

# 🧩 Evolution Flow

```text
Need Learning
      │
      ▼
Gradient Descent

Need Better Direction
      │
      ▼
Momentum

Need Adaptive Learning Rate
      │
      ▼
AdaGrad

Need to Forget Old Gradients
      │
      ▼
RMSProp

Need Direction + Adaptive LR
      │
      ▼
Adam

Need Better Generalization
      │
      ▼
AdamW
```

---

# 📌 Final Takeaways

```text
🟢 Gradient Descent
→ Learns.

🟡 Momentum
→ Remembers direction.

🔵 AdaGrad
→ Adapts learning rate.

🟣 RMSProp
→ Forgets outdated gradients.

🔴 Adam
→ Combines direction + adaptive learning rate.

🟠 AdamW
→ Adds independent weight control for better generalization.
```