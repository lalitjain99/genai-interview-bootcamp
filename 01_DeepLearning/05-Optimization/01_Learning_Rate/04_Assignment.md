# Learning Rate — Assignment

---

# Assignment 1: Choosing the Right Learning Rate

Suppose the gradient for a weight is

```text
∇L = 100
```

Three engineers suggest different learning rates:

```text
Engineer A: η = 0.0001

Engineer B: η = 0.01

Engineer C: η = 100
```

Without performing detailed calculations, answer the following:

1. Which learning rate is likely to make training extremely slow?
2. Which learning rate is likely to overshoot the minimum?
3. Which learning rate appears to be the most reasonable?
4. Explain your reasoning.

---

# Assignment 2: Slow vs Fast Learning

Two identical neural networks are trained on the same dataset.

Model A uses

```text
η = 0.00001
```

Model B uses

```text
η = 10
```

After 100 epochs,

neither model reaches the minimum.

Explain why both models failed,

even though their learning rates are completely different.

---

# Assignment 3: Convince Your Manager

Your manager says:

> "If a larger learning rate helps the model learn faster, why don't we always choose the largest possible learning rate?"

As the Machine Learning Engineer,

write a short explanation convincing your manager why this idea is incorrect.

---

# Assignment 4: Debug the Training

During training,

the loss behaves as follows:

```text
10

↓

8

↓

6

↓

5

↓

7

↓

12

↓

20
```

Answer the following:

1. What does this behaviour indicate?
2. What are the possible reasons behind it?
3. Should you immediately reduce the learning rate? Why or why not?

---

# Assignment 5: Think Like an Optimizer Designer

During one discussion,

a researcher says:

> "The gradient already contains magnitude. Why do we need another parameter called the learning rate?"

Without referring to any optimizer beyond Gradient Descent,

answer the following:

1. Why isn't the gradient alone sufficient?
2. What independent role does the learning rate play?
3. How does separating direction from step size make optimization more stable?

---

# Staff Engineer Challenge

You are leading the training of a Large Language Model.

Training has been stable for several hours.

Suddenly,

the loss behaves like this:

```text
10.2

↓

8.9

↓

7.8

↓

6.9

↓

15.1

↓

31.4

↓

NaN
```

Some engineers immediately suggest reducing the learning rate.

As the Staff Engineer,

describe your debugging strategy.

Consider questions such as:

- Is the learning rate really the root cause?
- What other factors could produce the same behaviour?
- What evidence would you gather before changing any hyperparameters?
- How would you systematically isolate the actual problem?

> **Hint:** Think like an engineer solving a production training issue, not like someone tuning a single hyperparameter.