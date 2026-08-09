# 📘 Lecture 08 — Residual Connections

> **Core Idea:** Residual Connections allow information and gradients to bypass one or more layers through a shortcut path. Instead of forcing a deep network to learn an entire transformation from scratch, the network learns the **difference**, or residual, that should be added to the original input.

---

# 🎯 Learning Objectives

By the end of this lecture, you will understand:

- What a Residual Connection is
- Why very deep neural networks become difficult to train
- What the degradation problem is
- The difference between degradation and overfitting
- The residual equation:

  `y = F(x) + x`

- What `F(x)` represents
- What the identity/shortcut path does
- Why learning a residual can be easier than learning the full mapping
- How Residual Connections improve gradient flow
- Why gradients get an additional direct path
- What happens when dimensions do not match
- What a projection shortcut is
- The structure of a Residual Block
- How Residual Connections relate to ResNet
- Why skip connections are now used far beyond CNNs

---

# 1️⃣ Why Do We Need Residual Connections?

A natural idea in deep learning is:

> If adding layers gives the model more representational power, then making the network deeper should always improve performance.

For example:

Input  
↓  
Layer 1  
↓  
Layer 2  
↓  
Layer 3  
↓  
Output

may be improved by:

Input  
↓  
Layer 1  
↓  
Layer 2  
↓  
Layer 3  
↓  
Layer 4  
↓  
Layer 5  
↓  
Output

In theory, the deeper model has at least as much representational capacity as the shallower one.

But in practice, researchers found something surprising:

> Simply adding more layers can make the deeper network **harder to optimize**.

Training error can actually become worse.

This leads to an important problem:

> **The Degradation Problem**

---

# 2️⃣ What Is the Degradation Problem?

Suppose we have:

### Network A

20 layers

### Network B

50 layers

You might expect:

50-layer network  
↓  
More capacity  
↓  
Lower training error

But sometimes we observe:

20-layer network  
↓  
Better training accuracy

50-layer network  
↓  
Worse training accuracy

This is called the:

> **Degradation Problem**

The deeper network has enough capacity, but optimization becomes more difficult.

---

# 3️⃣ Degradation Is NOT the Same as Overfitting

This distinction is extremely important.

## Overfitting

Overfitting means:

Training Performance → Good  
Validation Performance → Worse

The model memorizes the training data but generalizes poorly.

---

## Degradation

Degradation means:

Training Performance itself becomes worse as depth increases.

So:

More Layers  
↓  
Harder Optimization  
↓  
Higher Training Error

The model is not merely overfitting.

It is struggling to optimize the deeper architecture.

---

# 4️⃣ Why Should a Deeper Network At Least Match a Shallower One?

Imagine a 20-layer network already learns a good transformation.

Now we add another layer.

In theory, the extra layer could simply learn:

Identity Mapping

Meaning:

Output = Input

So the additional layer should theoretically not make the network worse.

Conceptually:

Existing Good Representation  
↓  
Extra Layer  
↓  
Do Nothing  
↓  
Same Representation

But learning an exact identity mapping through ordinary nonlinear layers is not always easy for an optimizer.

That is one of the motivations for Residual Connections.

---

# 5️⃣ What Is an Identity Mapping?

An identity mapping simply means:

Input goes in  
↓  
Same value comes out

Mathematically:

y = x

Example:

x = 5

Then:

y = 5

Nothing changes.

In a neural network:

Feature Representation x  
↓  
Identity Mapping  
↓  
Same Feature Representation x

Residual Connections make this identity path directly available.

---

# 6️⃣ What Is a Residual Connection?

A Residual Connection creates a shortcut around one or more layers.

Without a Residual Connection:

Input x  
↓  
Layer  
↓  
Layer  
↓  
Output F(x)

With a Residual Connection:

x ─────────────────────────┐
│                          │
↓                          │
Layer                      │
↓                          │
Layer                      │
↓                          │
F(x)                       │
│                          │
└──────── Add x ◄──────────┘
            ↓
        F(x) + x

The final output becomes:

y = F(x) + x

This is the fundamental residual equation.

---

# 7️⃣ What Does `F(x)` Mean?

`F(x)` represents the transformation learned by the layers inside the residual branch.

For example:

x  
↓  
Layer 1  
↓  
Activation  
↓  
Layer 2  
↓  
F(x)

So:

F(x)

is whatever transformation those layers learn.

The shortcut path directly carries:

x

Then the two are added:

y = F(x) + x

---

# 8️⃣ Why Is It Called "Residual" Learning?

Suppose the desired transformation is:

H(x)

A traditional neural network tries to directly learn:

H(x)

Residual learning instead rewrites this as:

H(x) = F(x) + x

Therefore:

F(x) = H(x) - x

So `F(x)` is the:

> **Residual**

or:

> The difference between the desired output and the original input.

---

# 9️⃣ Simple Numerical Example

Suppose:

Input:

x = 10

Desired output:

H(x) = 12

A normal network tries to learn:

10 → 12

Residual learning asks:

What change should be added to 10?

Residual:

F(x) = 12 - 10

F(x) = 2

Then:

y = F(x) + x

y = 2 + 10

y = 12

So instead of learning the entire output:

12

the residual branch only needs to learn:

+2

---

# 🔟 Why Can Learning the Residual Be Easier?

Suppose the desired mapping is close to the identity.

For example:

Input:

x = 100

Desired output:

H(x) = 101

A standard block must learn:

H(x) = 101

Residual block learns:

F(x) = H(x) - x

F(x) = 101 - 100

F(x) = 1

So the residual branch only learns:

Small Correction

rather than reconstructing the entire representation.

Conceptually:

Original Representation  
+  
Small Useful Change  
↓  
Improved Representation

---

# 1️⃣1️⃣ What If No Transformation Is Needed?

This is one of the strongest intuitions behind residual learning.

Suppose the ideal output is simply:

H(x) = x

Then:

F(x) = H(x) - x

So:

F(x) = x - x

F(x) = 0

Therefore:

y = F(x) + x

y = 0 + x

y = x

So if extra layers are unnecessary, the residual branch only needs to learn:

F(x) ≈ 0

The identity information already flows through the shortcut.

This makes it easier for deeper networks to behave like shallower networks when necessary.

---

# 1️⃣2️⃣ Plain Network vs Residual Network

## Plain Network

The layers must learn the full mapping:

x  
↓  
Layers  
↓  
H(x)

---

## Residual Network

The layers learn only the correction:

x ────────────────────┐
│                     │
↓                     │
Layers                 │
↓                     │
F(x)                   │
│                     │
└────── Add x ◄────────┘
           ↓
       H(x)

where:

H(x) = F(x) + x

---

# 1️⃣3️⃣ Why Are Residual Connections Useful in Very Deep Networks?

As networks become deeper:

More Layers  
↓  
More Transformations  
↓  
More Difficult Optimization  
↓  
Harder Gradient Flow

Residual Connections provide:

Shortcut Paths

through the network.

So information does not have to pass only through every transformation.

Instead:

Information can flow:

Through Layers

AND

Through Shortcut

This gives the network multiple paths.

---

# 1️⃣4️⃣ Residual Connections and Gradient Flow

This is one of the most important reasons Residual Connections work so well.

Suppose:

y = F(x) + x

During backpropagation:

We want:

∂Loss / ∂x

Using the Chain Rule:

∂Loss / ∂x
=
∂Loss / ∂y
×
∂y / ∂x

Now:

y = F(x) + x

Therefore:

∂y / ∂x
=
∂F(x) / ∂x + 1

That `+1` is extremely important.

---

# 1️⃣5️⃣ Why Is the `+1` Important?

Without a shortcut:

y = F(x)

So:

∂y / ∂x
=
∂F(x) / ∂x

The gradient depends completely on the derivative through the learned layers.

If that derivative becomes very small:

Gradient  
↓  
Small  
↓  
Smaller  
↓  
Vanishing Risk

---

With the residual connection:

y = F(x) + x

So:

∂y / ∂x
=
∂F(x) / ∂x + 1

Now there is a direct identity contribution.

Conceptually:

Gradient
   │
   ├──────────────→ Through F(x)
   │
   └──────────────→ Identity Path
                         ↓
                         1

So gradients have another route through the network.

---

# 1️⃣6️⃣ Simple Gradient Example

Suppose:

∂F(x) / ∂x = 0.01

Without residual connection:

Gradient multiplier:

0.01

So:

Gradient  
↓  
× 0.01  
↓  
Very Small

---

With residual connection:

∂y / ∂x
=
0.01 + 1

=
1.01

So there is a strong direct path for gradient flow.

This does not mean gradients can never vanish.

But it makes optimization much easier.

---

# 1️⃣7️⃣ Important Clarification About the Gradient

Do not interpret:

∂F(x) / ∂x + 1

as meaning:

> "The gradient is always exactly 1."

That is incorrect.

The full gradient still depends on:

- Upstream gradients
- The learned residual function
- Other network operations
- Multiple residual blocks

The important point is:

> **The shortcut introduces a direct derivative path that does not need to pass through all transformations in the residual branch.**

---

# 1️⃣8️⃣ Information Flow vs Gradient Flow

Residual Connections help in two related ways.

## Forward Pass

Input information can bypass unnecessary transformations.

x  
↓  
Shortcut  
↓  
Later Representation

---

## Backward Pass

Gradients can propagate through the shortcut path.

Loss  
↓  
Backward Gradient  
↓  
Shortcut  
↓  
Earlier Layers

So Residual Connections improve:

Forward Information Flow  
+  
Backward Gradient Flow

---

# 1️⃣9️⃣ What Is a Residual Block?

A group of layers wrapped by a shortcut connection is called a:

> **Residual Block**

Conceptually:

Input x  
│
├──────────────────────────────┐
│                              │
↓                              │
Layer                          │
↓                              │
Activation                     │
↓                              │
Layer                          │
↓                              │
F(x)                           │
│                              │
└──────────── Add ◄────────────┘
                 ↓
             Output

# 2️⃣0️⃣ Identity Shortcut

When the input `x` can be added directly to `F(x)` without any transformation:

`y = F(x) + x`

the shortcut is called an:

> **Identity Shortcut**

because the shortcut simply passes:

`x`

unchanged.

Conceptually:

x  
↓  
Shortcut  
↓  
x

So:

Shortcut(x) = x

---

# 2️⃣1️⃣ Why Must Dimensions Match?

To add:

`F(x) + x`

both tensors must have compatible shapes.

For example:

x shape:

`[batch, 64]`

F(x) shape:

`[batch, 64]`

Then addition works.

But suppose:

x:

`[batch, 64]`

F(x):

`[batch, 128]`

Then:

`F(x) + x`

cannot be directly calculated.

The dimensions do not match.

---

# 2️⃣2️⃣ What Do We Do When Dimensions Don't Match?

We transform the shortcut.

Instead of:

`y = F(x) + x`

we use:

`y = F(x) + Wₛx`

where:

`Wₛ`

is a learned projection.

This is often called a:

> **Projection Shortcut**

---

# 2️⃣3️⃣ Projection Shortcut Example

Suppose:

x has 64 features

but:

F(x) has 128 features

We need:

64  
↓  
Projection  
↓  
128

Then:

Residual Branch:

x  
↓  
Layers  
↓  
F(x) with 128 features

Shortcut Branch:

x  
↓  
Projection Wₛ  
↓  
128 features

Now:

`F(x) + Wₛx`

can be added.

---

# 2️⃣4️⃣ Identity vs Projection Shortcut

| Identity Shortcut | Projection Shortcut |
|---|---|
| `x` passed directly | `x` transformed |
| No extra learned weights | Contains learned weights |
| Used when dimensions match | Used when dimensions differ |
| `y = F(x) + x` | `y = F(x) + Wₛx` |

---

# 2️⃣5️⃣ Residual Connection Does NOT Mean Concatenation

This is important.

Residual Connections generally use:

> **Addition**

Example:

`y = F(x) + x`

They do not usually mean:

Concatenate `F(x)` and `x`

Addition and concatenation are different operations.

### Addition

Same dimensional representation:

`F(x) + x`

### Concatenation

Join features:

`[F(x), x]`

Residual learning is primarily based on addition.

---

# 2️⃣6️⃣ Why Addition?

Addition allows the shortcut and learned transformation to represent:

Original Information  
+  
Correction

That directly gives us:

`H(x) = x + F(x)`

which is the residual-learning formulation.

It also preserves the representation size when dimensions match.

---

# 2️⃣7️⃣ Does the Shortcut Have Trainable Parameters?

For an identity shortcut:

`y = F(x) + x`

the shortcut itself typically has:

> **No trainable parameters**

It simply passes `x`.

For a projection shortcut:

`y = F(x) + Wₛx`

the projection:

`Wₛ`

contains trainable parameters.

---

# 2️⃣8️⃣ How Residual Connections Address Degradation

Recall the degradation problem:

Deeper Network  
↓  
Harder Optimization  
↓  
Higher Training Error

Residual Connections make it easier for extra layers to behave like identity transformations.

If the new layers are unnecessary:

`F(x) ≈ 0`

Then:

`F(x) + x`

≈

`x`

So adding more residual blocks does not force the network to completely relearn existing useful representations.

---

# 2️⃣9️⃣ Residual Connection Does NOT Mean "Skip These Layers"

This is another common misunderstanding.

The layers are still used.

We compute:

`F(x)`

through the residual branch.

Then:

`F(x)`

is combined with:

`x`

The network can learn how much correction should be produced.

So the shortcut does not simply disable the layers.

---

# 3️⃣0️⃣ What Happens During Training?

At the beginning:

Residual branch  
↓  
Randomly initialized transformation

Shortcut  
↓  
Passes original representation

As training progresses:

Residual branch learns:

`F(x)`

which represents useful modifications.

So the block gradually learns:

Output  
=  
Original Information  
+  
Learned Improvement

---

# 3️⃣1️⃣ Residual Connection and Vanishing Gradients

Residual connections do not completely eliminate vanishing gradients.

But they provide a shorter gradient path.

Without residuals:

Loss  
↓  
Layer  
↓  
Layer  
↓  
Layer  
↓  
Layer  
↓  
Early Layer

With residuals:

Loss  
↓
┌─────────────────────────┐
│                         │
Layers                    Shortcut
│                         │
└────────────┬────────────┘
             ↓
        Earlier Layer

This makes gradient propagation easier.

---

# 3️⃣2️⃣ Residual Connections and Exploding Gradients

Residual connections are mainly known for making deep-network optimization easier and improving gradient flow.

They do not automatically guarantee that exploding gradients cannot occur.

You may still need:

- Proper initialization
- Normalization
- Appropriate learning rate
- Suitable optimizer
- Gradient clipping in some situations

So:

Residual Connections  
≠  
Complete Solution to Every Optimization Problem

---

# 3️⃣3️⃣ How Residual Connections Relate to Weight Initialization

This connects directly with our previous module.

Good Weight Initialization:

Healthy initial signal scale

Residual Connections:

Provide shorter information and gradient paths

Together:

Good Initialization  
+  
Residual Connections  
↓  
Much Easier Deep-Network Optimization

These techniques solve different but complementary problems.

---

# 3️⃣4️⃣ Residual Connections and Batch Normalization

Residual blocks are often combined with normalization techniques.

Conceptually:

Layer  
↓  
Normalization  
↓  
Activation  
↓  
Layer  
↓  
Add Shortcut

The exact ordering can differ across architectures.

For now, the important point is:

> Residual Connections and normalization are complementary techniques for stabilizing deep-network training.

---

# 3️⃣5️⃣ ResNet

Residual Connections became especially famous through:

> **ResNet — Residual Network**

The key idea was to organize very deep networks into Residual Blocks.

Instead of:

Layer  
↓  
Layer  
↓  
Layer  
↓  
Layer  
↓  
...

we get:

Residual Block  
↓  
Residual Block  
↓  
Residual Block  
↓  
Residual Block  
↓  
...

Each block contains shortcut connections.

This made training much deeper networks practical.

---

# 3️⃣6️⃣ Basic Residual Block Concept

A simplified block:

Input x  
│
├───────────────────────────────┐
│                               │
↓                               │
Linear / Convolution Layer      │
↓                               │
Activation                      │
↓                               │
Linear / Convolution Layer      │
↓                               │
F(x)                            │
│                               │
└─────────────── Add x ◄────────┘
                    ↓
                 Output

We do not need CNN details yet.

The same mathematical idea applies to ordinary neural-network layers.

---

# 3️⃣7️⃣ Residual Connections Beyond CNNs

Residual Connections are not limited to image models.

The same idea appears throughout modern deep learning.

Conceptually:

Input  
↓  
Transformation  
↓  
+ Input  
↓  
Output

This pattern is useful anywhere very deep transformations need stable information and gradient flow.

So residual/skip connections became a general architectural principle.

---

# 3️⃣8️⃣ Residual Connection vs Skip Connection

These terms are often used closely.

### Skip Connection

A broad term meaning:

> Information skips one or more layers.

### Residual Connection

A common type of skip connection where:

`y = F(x) + x`

So:

Residual Connection  
⊂  
Skip Connections

Not every possible skip connection must necessarily use residual addition, but residual addition is one of the most important forms.

---

# 3️⃣9️⃣ Why Is Identity Mapping Easy with Residual Connections?

Suppose:

Desired Mapping:

`H(x) = x`

Without residual:

Layers must directly learn:

`H(x) = x`

With residual:

`H(x) = F(x) + x`

So:

`x = F(x) + x`

Therefore:

`F(x) = 0`

The residual branch only needs to produce values near zero.

This is often easier than forcing several nonlinear transformations to exactly reproduce the original input.

---

# 4️⃣0️⃣ Residual Learning Example

Suppose the network receives a representation:

`x = 8`

and the useful next representation should be:

`8.5`

Without residual learning:

Network must learn:

8 → 8.5

With residual learning:

`F(x) = 0.5`

Then:

Output  
=  
8 + 0.5  
=  
8.5

You can think of the residual branch as learning:

> **What needs to change?**

instead of:

> **What should the entire representation become?**

---

# 4️⃣1️⃣ Practical Mental Model

Imagine you already have a good document.

A normal deep layer approach says:

> Rewrite the entire document.

Residual learning says:

> Keep the existing document and only write the corrections.

So:

Existing Information  
+  
Useful Corrections  
↓  
Improved Information

This is the central intuition.

---

# 4️⃣2️⃣ Residual Connections and Feature Reuse

Because the shortcut preserves `x`, later layers can reuse earlier representations.

Instead of throwing away all previous information:

x  
↓  
Completely Replace

we get:

x  
+  
Learned Modification

So useful features can continue to flow deeper into the network.

---

# 4️⃣3️⃣ What If the Residual Is Large?

Residual learning does not require:

`F(x)`

to always be small.

The network is free to learn a large transformation when necessary.

The idea is simply that it has the option to represent the output as:

Original Representation  
+  
Learned Change

If a large change is needed:

`F(x)`

can become large.

---

# 4️⃣4️⃣ Does Residual Connection Reduce the Number of Layers?

No.

Residual Connections allow us to train **more layers effectively**.

They do not necessarily make the architecture shallow.

In fact:

Residual Connections  
↓  
Easier Optimization  
↓  
Much Deeper Networks Become Practical

---

# 4️⃣5️⃣ Does Residual Connection Reduce Model Parameters?

Not necessarily.

An identity shortcut adds virtually no parameters.

But the residual branch still contains its layers.

Projection shortcuts can also add parameters.

So the main purpose is not parameter reduction.

The main purpose is:

> **Better optimization and signal flow**

---

# 4️⃣6️⃣ Does Residual Connection Prevent Overfitting?

Not directly.

Residual Connections mainly address:

- Optimization difficulty
- Degradation
- Information flow
- Gradient flow

Overfitting is a generalization problem.

It may require:

- Regularization
- Data augmentation
- Early stopping
- Weight decay
- Other techniques

Do not confuse:

Residual Connections  
→ Optimization

with:

Regularization  
→ Generalization

---

# 4️⃣7️⃣ Plain Network vs Residual Network Summary

| Plain Network | Residual Network |
|---|---|
| Learns full transformation | Learns residual correction |
| `y = F(x)` | `y = F(x) + x` |
| Information must pass through layers | Shortcut provides direct path |
| Deep optimization can be difficult | Easier deep optimization |
| Identity mapping can be difficult | Identity mapping is naturally available |
| Gradient has one transformed path | Gradient has shortcut path too |

---

# 4️⃣8️⃣ Residual Block When Dimensions Match

If:

`Shape(x) = Shape(F(x))`

use:

`y = F(x) + x`

This is the:

> **Identity Shortcut**

---

# 4️⃣9️⃣ Residual Block When Dimensions Differ

If:

`Shape(x) ≠ Shape(F(x))`

we need to transform the shortcut:

`y = F(x) + Wₛx`

This is the:

> **Projection Shortcut**

where:

`Wₛ`

changes the input representation to a compatible shape.

---

# 5️⃣0️⃣ Complete Residual Connection Flow

Very Deep Network  
↓  
Optimization Becomes Hard  
↓  
Degradation Problem  
↓  
Need Easier Identity / Information Flow  
↓  
Residual Connection  
↓  
`y = F(x) + x`  
↓  
Residual Branch Learns Correction  
+  
Shortcut Preserves Input  
↓  
Forward Information Flows More Easily  
+  
Backward Gradient Gets Direct Path  
↓  
Very Deep Networks Become Easier to Train

---

# 🎤 Interview Perspective

If an interviewer asks:

> **What is a Residual Connection and why is it useful?**

A strong answer is:

> **A Residual Connection is a shortcut connection where the input is added directly to the output of one or more transformed layers, typically written as `y = F(x) + x`. Instead of forcing the layers to learn the complete mapping, they learn a residual correction `F(x)`. The shortcut also provides a direct path for information and gradients, making very deep networks easier to optimize and helping address the degradation problem. If input and output dimensions differ, a projection such as `Wₛx` can be used on the shortcut path.**

---

# 🎯 30-Second Explanation

Residual learning asks:

> **What should change about the input?**

instead of:

> **What should the entire output be?**

Mathematically:

Desired Mapping:

`H(x)`

Residual:

`F(x) = H(x) - x`

Therefore:

`H(x) = F(x) + x`

The shortcut carries:

`x`

while the residual branch learns:

`F(x)`

So:

Original Information  
+  
Learned Correction  
↓  
Final Representation

---

# 🧠 Complete Mental Model

Input x
│
├─────────────────────────────────┐
│                                 │
↓                                 │
Learned Layers                    │
↓                                 │
F(x)                              │
│                                 │
└────────────── Add x ◄───────────┘
                    ↓
                 Output
                    ↓
              F(x) + x


Forward:

Original Information  
+  
Learned Correction


Backward:

Gradient Through Residual Branch  
+  
Gradient Through Identity Shortcut

                    ↓

          Easier Deep Optimization

---

# 📌 Key Takeaways

- Very deep plain networks can suffer from the **degradation problem**
- Degradation is different from overfitting
- A Residual Connection creates a shortcut around layers
- The basic residual equation is:

  `y = F(x) + x`

- `x` is the input to the current residual block
- `x` can be the raw network input or a representation from earlier layers
- `F(x)` is the learned residual transformation
- We do **not** know the ideal intermediate `H(x)` explicitly
- `H(x)` is conceptual notation for whatever useful mapping the block should ultimately represent
- Training learns `F(x)` indirectly through the final loss and backpropagation
- If the existing representation is already useful, the residual branch can learn:

  `F(x) ≈ 0`

- Residual Connections provide better forward information flow
- They also provide an additional path for backward gradients
- For:

  `y = F(x) + x`

  the derivative includes:

  `∂F(x)/∂x + 1`

- The `+1` represents the identity gradient path
- Residual Connections reduce optimization difficulty in very deep networks
- If dimensions match, use an identity shortcut
- If dimensions differ, use a projection shortcut:

  `y = F(x) + Wₛx`

- Residual addition is different from concatenation
- Residual Connections do not directly solve overfitting
- They do not guarantee gradients can never vanish or explode
- Residual block depth and width are architecture choices
- The network learns during training how much each residual block should actually modify its input
- Residual Connections are complementary to proper initialization and normalization
- ResNet popularized Residual Connections
- Skip connections are now a fundamental idea throughout deep learning

---

# ⭐ Golden Rule

> **A Residual Connection preserves what is already useful and asks the network to learn only what needs to change.**

The equation to remember is:

`y = F(x) + x`

And the intuition is:

Original Representation  
+  
Learned Correction  
↓  
Better Representation