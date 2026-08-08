# 📘 Lecture 02 — Zero Initialization

# 🎯 Learning Objectives

By the end of this lecture, you will understand:

- What Zero Initialization is.
- Why it looks like a reasonable idea.
- Why it completely fails in deep neural networks.
- What is the Symmetry Problem.
- Why neurons must learn different features.
- Why Zero Initialization works for Logistic Regression but not Neural Networks.

---

# 🤔 What is Zero Initialization?

Zero Initialization means assigning **every weight** in the neural network the value **0** before training begins.

For example,

```
Input Layer          Hidden Layer

x₁ ──(0)──► h₁

x₂ ──(0)──► h₁

x₃ ──(0)──► h₁
```

Every connection weight starts as

```
w = 0
```

---

# 💭 At First Glance, It Seems Logical

Many beginners think,

> "If all neurons start equally, won't Gradient Descent gradually learn different weights?"

It sounds reasonable.

Unfortunately,

this intuition is **wrong**.

---

# 🎯 Let's Understand Using a Small Network

Suppose we have

```
          Hidden Layer

        h₁       h₂
       /  \     /  \
x₁ ───┘    \   /    └── x₂
```

Initially,

```
w₁ = 0

w₂ = 0

w₃ = 0

w₄ = 0
```

Both hidden neurons have **exactly identical weights**.

---

# ▶️ Forward Propagation

Recall,

```
z = wx + b
```

Suppose

```
Input

x₁ = 2

x₂ = 3
```

Since every weight is zero,

```
z₁ = 0

z₂ = 0
```

After the activation function,

```
a₁ = f(0)

a₂ = f(0)
```

Both neurons produce the **same output**.

```
Neuron 1

Output = 0.5 (Sigmoid)

Neuron 2

Output = 0.5
```

They behave identically.

---

# 🔄 Backpropagation

Now the model computes gradients.

Remember,

gradients depend on:

- Inputs
- Outputs
- Current Weights

Since

- both neurons received the same input,
- both produced the same output,
- both have the same weights,

they also receive the **same gradients**.

```
dw₁ = dw₂
```

---

# 🔄 Gradient Descent

Gradient Descent updates weights as

```
w = w − η × dw
```

Suppose

```
dw = 0.2

Learning Rate = 0.1
```

Both neurons become

```
w₁ = -0.02

w₂ = -0.02
```

Still identical.

---

# 🔁 Next Iteration

The next forward pass again produces

```
Same Inputs

↓

Same Outputs

↓

Same Gradients

↓

Same Weight Updates
```

Nothing changes.

Both neurons remain exact copies of each other throughout training.

---

# 🚨 This is Called the Symmetry Problem

Because every neuron starts identically,

they continue learning identically.

```
Neuron 1

↓

Learns Feature A

Neuron 2

↓

Also Learns Feature A

Neuron 3

↓

Also Learns Feature A
```

Instead of learning different patterns,

every neuron becomes a duplicate.

---

# 🧠 Why Is This a Problem?

The power of neural networks comes from **different neurons learning different features**.

For example,

in an image classifier,

```
Neuron 1

↓

Edges

Neuron 2

↓

Corners

Neuron 3

↓

Textures

Neuron 4

↓

Shapes
```

Each neuron contributes unique information.

With Zero Initialization,

every neuron learns the **same feature**.

The hidden layer behaves as if it contains only **one neuron**.

---

# 🎨 Real-Life Analogy

Imagine five students solving the same problem.

### Good Scenario

Each student thinks independently.

```
Student 1 → Idea A

Student 2 → Idea B

Student 3 → Idea C
```

The team benefits from diverse ideas.

---

### Zero Initialization

Now imagine every student copies the first student's answer.

```
Student 1 → Idea A

Student 2 → Idea A

Student 3 → Idea A

Student 4 → Idea A
```

Having five students adds no value.

This is exactly what happens with Zero Initialization.

---

# ❓ Why Is It Called "Breaking Symmetry"?

Initially,

all neurons are perfectly identical.

```
Weights

0

0

0

0
```

Random Initialization introduces tiny differences.

Example,

```
0.003

-0.001

0.002

-0.004
```

Now every neuron starts from a slightly different point.

These tiny differences grow during training,

allowing neurons to learn different features.

This process is called **Symmetry Breaking**.

---

# ⚠️ Does Zero Initialization Always Fail?

No.

It depends on the model.

---

## Logistic Regression

Logistic Regression has **only one output neuron**.

```
Inputs

↓

Single Output
```

There are **no multiple hidden neurons**.

Therefore,

there is **no symmetry problem**.

Zero Initialization works perfectly fine.

---

## Neural Networks

Neural networks have many hidden neurons.

```
Inputs

↓

Hidden Layer

↓

Output
```

If all hidden neurons start identically,

they remain identical.

Therefore,

Zero Initialization fails.

---

# 📊 Logistic Regression vs Neural Network

| Logistic Regression | Neural Network |
|---------------------|----------------|
| One output neuron | Many hidden neurons |
| No symmetry problem | Symmetry problem |
| Zero Initialization works | Zero Initialization fails |

---

# 🚀 Key Takeaways

- Zero Initialization assigns every weight the value **0**.
- During forward propagation, all neurons produce identical outputs.
- During backpropagation, all neurons receive identical gradients.
- Gradient Descent updates every neuron in exactly the same way.
- Hidden neurons never specialize and become duplicates.
- This is known as the **Symmetry Problem**.
- Neural networks require **symmetry breaking**, which is achieved through Random Initialization.
- Logistic Regression can safely use Zero Initialization because it has only one output neuron.