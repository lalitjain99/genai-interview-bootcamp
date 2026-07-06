Lecture 1 : Perceptron — The Birth of Neural Networks
Learning Objectives

By the end of this lecture, you should be able to answer:

Why Linear Regression cannot solve AI problems.
Why Feature Engineering became the bottleneck.
What is Representation Learning.
What is a Perceptron.
Components of a Perceptron.
Why Perceptron is still linear.
Why Hidden Layers are required.
Why Deep Learning started.
Historical Motivation

Before Deep Learning,

Machine Learning looked like this:

Raw Data

↓

Feature Engineering (Human)

↓

Linear Model

↓

Prediction

Example

Face Recognition

Engineer manually extracts

Nose Length
Eye Distance
Skin Color
Face Width

Then

Linear Regression

Decision Tree

SVM

etc.

Problem?

Human has to decide what features matter.

Representation Learning

Deep Learning changed everything.

Instead of

Human

↓

Features

↓

Model

We now have

Raw Data

↓

Neural Network

↓

Features (Automatically Learned)

↓

Prediction

The network itself discovers useful features.

This is called

Representation Learning

Mathematical Model

Perceptron computes

z=w
1
	​

x
1
	​

+w
2
	​

x
2
	​

+⋯+w
n
	​

x
n
	​

+b

Then

Step Function

f(z)={
1
0
	​

z>0
z≤0
	​


Output

0 or 1

Components

Input

↓

Weights

↓

Weighted Sum

↓

Bias

↓

Activation

↓

Prediction

Weight

Weight determines

Direction of influence
Magnitude of influence

Positive Weight

Increase feature

↓

Increase output

Negative Weight

Increase feature

↓

Decrease output

Bias

Without bias

Every decision boundary passes through origin.

Bias allows

Decision Boundary

to shift anywhere.

Without Bias

Many simple functions become impossible to learn.

Activation

Perceptron uses

Step Function

Purpose

Convert continuous weighted sum

↓

Binary Decision

Why Perceptron is still Linear

Even though

Step Function

looks nonlinear,

Decision Boundary remains

One Hyperplane

Therefore

Cannot solve

XOR

Circles

Complex Images

Language

etc.

Limitations

❌ Needs Feature Engineering

❌ Linear Decision Boundary

❌ Cannot solve XOR

❌ Step Function not differentiable

❌ Cannot train using Gradient Descent

Key Takeaways

Perceptron introduced

Weights
Bias
Activation

These ideas still exist inside GPT today.

Only activation function changed.
