# Lecture 3 — Activation Functions
# Part 1: Why Neural Networks Need Activation Functions

---

# 🎯 Learning Objectives

By the end of this lecture you should be able to answer:

- Why is the weighted sum not enough?
- Why do we need activation functions?
- What problem were activation functions trying to solve?
- What is a Step Function?
- Why was it revolutionary in 1958?
- Why is it no longer used?
- What properties should an ideal activation function have?
- How did activation functions shape the future of Deep Learning?

---

# Chapter 1 — The Perceptron Can Calculate, But Can It Decide?

Let's continue our story.

So far we've built an Artificial Neuron.

It receives inputs.

It multiplies each input by a weight.

It adds a bias.

Finally it computes

\[
z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
\]

Looks impressive.

But let's ask a simple question.

---

Suppose we are building an Email Spam Classifier.

The model receives the following email.

```
Subject:
Congratulations!

Body:
You have won ₹10,00,000.
Click here immediately.
```

After computation, the neuron produces

```
z = 8.42
```

Now pause.

What exactly does **8.42** mean?

Is the email spam?

Is it safe?

Should Gmail block it?

Should it send it to Promotions?

Should it ignore it?

Nobody knows.

The neuron has produced a **score**.

But it has not produced a **decision**.

---

Imagine you go to a doctor.

After all the tests, the doctor says

```
Your report is...

73.2
```

You would immediately ask

> 73.2 what?

Blood pressure?

Sugar?

Heart rate?

Cancer probability?

The number itself has no meaning.

It needs interpretation.

Exactly the same problem exists inside a Perceptron.

The weighted sum is only a score.

The neuron still doesn't know what to do with it.

---

# Chapter 2 — The Missing Piece

Researchers in the late 1950s realized something.

A neuron shouldn't only calculate.

It should also **decide**.

Instead of returning

```
8.42
```

the neuron should return

```
Spam
```

or

```
Not Spam
```

It needed something that could convert a continuous numerical score into a meaningful decision.

That missing component became known as an **Activation Function**.

---

## The Neural Network Pipeline

Without an activation function, the neuron looks like this.

```
Input Features
       │
       ▼
Weighted Sum
(z = Wx + b)
       │
       ▼
     Output
```

The output is simply another number.

Now introduce an activation function.

```
Input Features
       │
       ▼
Weighted Sum
(z = Wx + b)
       │
       ▼
Activation Function
       │
       ▼
Prediction
```

The activation function acts as a **decision maker**.

---

# Chapter 3 — Learning from Biology

Frank Rosenblatt wasn't trying to invent mathematics.

He was trying to imitate the human brain.

A biological neuron behaves surprisingly simply.

It continuously receives electrical signals.

Some signals are weak.

Some are strong.

The neuron collects all of them.

Once the total electrical signal crosses a certain threshold...

The neuron fires.

Otherwise...

Nothing happens.

Biologically,

the neuron behaves almost like an ON/OFF switch.

```
Enough signal?

      Yes
       │
       ▼
     Fire

Otherwise

Don't Fire
```

Researchers copied exactly this behaviour.

---

# Chapter 4 — The First Activation Function

This gave birth to the **Step Function**.

Mathematically,

\[
f(z)=
\begin{cases}
1, & z > 0 \\
0, & z \le 0
\end{cases}
\]

The rule couldn't be simpler.

If the weighted sum is positive,

return **1**.

Otherwise,

return **0**.

The neuron finally became capable of making a decision.

---

# Chapter 5 — The Security Guard Analogy

Imagine an office building.

At the entrance sits a security guard.

Every visitor receives a security score.

```
Visitor A

Score = 91
```

The security policy says

```
Score > 50

↓

Allow Entry
```

Now another visitor arrives.

```
Score = 42
```

The guard immediately rejects them.

Notice something interesting.

The guard doesn't care whether the score is

```
51

or

5000
```

Once the threshold is crossed,

the decision is exactly the same.

The Step Function behaves in exactly the same way.

It only asks one question.

> Has the threshold been crossed?

If yes,

return **1**.

Otherwise,

return **0**.

---

# Chapter 6 — Why Was This Revolutionary?

Today,

the Step Function feels almost trivial.

But imagine the year is 1958.

Computers could execute instructions.

They could perform arithmetic.

But they couldn't make learned decisions.

For the first time,

a machine could

- receive data,
- compute a score,
- and automatically decide between two outcomes.

This was a historic breakthrough.

The Perceptron became the first learning algorithm inspired by the human brain.

It marked the birth of Neural Networks.

---

# Chapter 7 — The Hidden Limitation

Everything looked promising.

Until researchers tried solving harder problems.

Consider XOR.

```
A   B   XOR

0   0    0

0   1    1

1   0    1

1   1    0
```

No matter how the weights were adjusted,

the Perceptron failed.

Why?

Because the Step Function only converts the weighted sum into a binary decision.

It **does not** make the underlying model more expressive.

The decision boundary is still

\[
Wx + b = 0
\]

which is always a straight line (or hyperplane in higher dimensions).

Complex relationships require curved decision boundaries.

The Step Function cannot create those.

---

# Chapter 8 — An Even Bigger Problem

Researchers believed

"Perhaps we simply need to train the weights better."

That led to a new question.

Suppose the prediction is wrong.

How should we change the weights?

To answer that,

we need to know

> If I slightly change a weight,
> how much does the output change?

This is exactly what a derivative measures.

Now examine the Step Function.

On the negative side,

the output is always zero.

The slope is zero.

On the positive side,

the output is always one.

Again,

the slope is zero.

At the threshold,

the graph jumps abruptly.

The derivative does not even exist.

Therefore,

the Step Function provides no useful learning signal.

Without gradients,

there can be no Gradient Descent.

Without Gradient Descent,

there can be no Backpropagation.

Without Backpropagation,

modern Deep Learning becomes impossible.

---

# Chapter 9 — What Should an Ideal Activation Function Look Like?

The failure of the Step Function taught researchers an important lesson.

An activation function should satisfy several properties.

✅ It should introduce non-linearity.

✅ It should be differentiable.

✅ It should allow gradients to flow.

✅ It should be computationally efficient.

✅ It should train reliably in deep networks.

The search for such a function eventually led to

- Sigmoid
- Tanh
- ReLU
- GELU

Each one solved some problems while introducing new ones.

That evolution shaped the history of Deep Learning.

---

# 🤖 Engineering Connection

Imagine replacing every GELU activation inside GPT with the Step Function.

The model could still perform a forward pass.

However,

during training,

every gradient would become zero or undefined.

The optimizer would have no meaningful direction to update billions of parameters.

Training would effectively stop.

This is why differentiable activation functions are one of the foundational requirements of modern Deep Learning.

---

# 🎯 Key Takeaways

- The weighted sum produces only a numerical score.
- Activation functions convert scores into meaningful outputs.
- The Step Function was inspired by biological neurons.
- It introduced binary decision making into neural networks.
- It still produces only linear decision boundaries.
- Its derivative is zero almost everywhere.
- Because of this, it cannot be trained using Gradient Descent.
- The search for better activation functions led to Sigmoid, ReLU and eventually GELU.

---

# 📖 Next Chapter

The Step Function taught us how neurons make decisions.

Now researchers asked a new question.

> Can we design an activation function that behaves like the Step Function...
>
> but is smooth enough to be trained?

The answer was **Sigmoid**.

And that single function restarted the Deep Learning revolution.