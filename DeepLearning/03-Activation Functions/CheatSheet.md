# Activation Functions Cheatsheet

## Step Function

Output:
0 or 1

Pros
- Simple

Cons
- Not differentiable
- Cannot use Gradient Descent

---

## Sigmoid

Equation
σ(x)=1/(1+e^-x)

Output
0–1

Pros
- Smooth
- Differentiable

Cons
- Vanishing Gradient

---

## ReLU

Equation

f(x)=max(0,x)

Pros
- Fast
- Sparse Activation
- Gradient = 1 for positive inputs

Cons
- Dying ReLU

---

## Leaky ReLU

Negative side slope = α

Pros
- Reduces Dying ReLU

Cons
- Still piecewise linear

---

## GELU

Idea

Output = Input × Probability(Input is useful)

Pros
- Smooth
- Used in GPT

Cons
- More computationally expensive

---

GPT uses

FFN

↓

Linear

↓

GELU

↓

Linear