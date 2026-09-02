# 📘 Lecture — CNN Feature Hierarchy

> **Core Idea:** CNNs do not jump directly from pixels to objects. They build understanding gradually: **simple patterns first, then combinations of patterns, then larger and more meaningful structures.**

---

# 🎯 1. Where We Are So Far

So far, we have learned:

```text
Image
↓
Convolution
↓
Filters detect patterns
↓
Feature Maps store responses
↓
Stride / Padding control geometry
↓
Multiple Filters create multiple channels
↓
Pooling reduces spatial size
↓
Parameter Sharing reuses filters
↓
Receptive Field tells how much input context one activation can see
```

Now an important question appears:

> If early layers only look at small local regions, how does a CNN eventually recognize something complex like a face, car, dog, or building?

The answer is:

> **Feature Hierarchy**

---

# 🧠 2. What Is a Feature?

A feature is simply:

> **some useful pattern in the input that helps the model solve its task**

For an image, features could include:

* edges
* corners
* curves
* textures
* repeated patterns
* shapes
* object parts
* larger structures

But remember:

> CNNs are not explicitly told, “detect edges” or “detect eyes.”

They learn filters through training that become useful for minimizing the loss.

---

# 🪜 3. What Does “Hierarchy” Mean?

Hierarchy means:

> **building complex representations from simpler ones**

Think of it like language:

```text
Letters
↓
Words
↓
Phrases
↓
Sentences
↓
Meaning
```

In CNNs:

```text
Pixels
↓
Simple local patterns
↓
Pattern combinations
↓
Shapes / textures
↓
Object parts
↓
Higher-level representations
```

This gradual construction is the:

> **CNN Feature Hierarchy**

---

# 🖼️ 4. Starting Point — Raw Pixels

Suppose we have an image of a cat.

The input may be:

`224 × 224 × 3`

Each spatial position contains RGB values.

At this stage, the network does not initially see:

> “cat”

It sees:

> numerical pixel values

For example:

```text
[124, 118, 110]
[130, 121, 114]
[220, 215, 208]
...
```

Raw pixels are the lowest-level representation.

---

# 🔍 5. First Convolutional Layers — Simple Features

Imagine the first layer has:

`64 filters`

Each filter may learn a different useful local pattern.

Because the receptive field is still relatively small, these filters mostly work with:

> local pixel relationships

Typical learned patterns may resemble:

* horizontal edges
* vertical edges
* diagonal changes
* color transitions
* simple gradients
* small texture patterns

Conceptually:

```text
Pixels
↓
Conv Layer 1
↓
Edges / Local Patterns
```

---

# ⚠️ Important Precision

Do not say:

> “The first layer always detects edges.”

That is too strict.

A better statement is:

> **Early CNN layers often learn relatively simple local features such as edges, gradients, color contrasts, and textures.**

What is actually learned depends on:

* dataset
* architecture
* task
* training

---

# 🎛️ 6. Why Early Layers Learn Simple Features

Why don't early layers directly detect a complete face?

Because their receptive field is small.

For example:

```text
First Conv
Kernel = 3×3
RF = 3×3
```

One activation can only access a tiny local region.

It cannot see:

* both eyes
* the nose
* the mouth
* overall head shape

So it naturally works with:

> **small local structures**

---

# 🧩 7. Second Layer Does Not See Raw Pixels Directly

This is one of the most important ideas.

The second convolution layer receives:

> **feature maps from the first layer**

not just the raw RGB image.

Example:

```text
Input Image
224×224×3
      ↓
Conv Layer 1
      ↓
224×224×64
      ↓
Conv Layer 2
```

Those 64 channels may represent responses to different learned local patterns.

So Conv Layer 2 can combine:

> **patterns discovered by Conv Layer 1**

---

# 🧠 8. Deeper Filters Combine Earlier Features

Suppose an early layer has learned filters responding strongly to:

* vertical edges
* horizontal edges
* diagonal edges
* curves

A deeper filter spans:

> all those input feature channels

and can learn combinations such as:

```text
vertical edge
+
horizontal edge
↓
corner-like structure
```

or:

```text
several curved responses
↓
circular / rounded structure
```

So:

> **deeper features are constructed from combinations of earlier features**

---

# 🧱 9. Feature Composition

Consider this progression:

```text
Pixels
↓
Edges
↓
Corners / Curves
↓
Textures / Small Shapes
↓
Object Parts
↓
Complex Object Representations
```

This is called:

> **compositional representation**

Complex features are built from simpler ones.

---

# 👁️ 10. Example — Face Recognition

Imagine a CNN trained on face images.

A simplified conceptual hierarchy might look like:

### Layer 1

Detects:

* edges
* brightness transitions
* color contrasts

### Layer 2–3

Combines them into:

* curves
* corners
* simple textures

### Middle Layers

May respond to:

* eye-like structures
* nose contours
* mouth-like shapes
* hair textures

### Deeper Layers

May respond to combinations such as:

* arrangement of facial parts
* larger face-like patterns

So:

```text
Pixels
↓
Edges
↓
Curves
↓
Eye-like / Nose-like patterns
↓
Face-level representation
```

---

# ⚠️ But Does One Neuron Equal “Eye Detector”?

Not necessarily.

This is another common oversimplification.

It is tempting to say:

> “Neuron 354 detects eyes.”

Sometimes units can have interpretable responses, but representations are often:

> **distributed across many channels and activations**

A deeper representation may encode useful information jointly across many feature maps.

So instead of saying:

> “This neuron is definitely an eye detector”

a safer statement is:

> **Some deeper activations may respond strongly to patterns correlated with object parts or higher-level structures.**

---

# 🗺️ 11. Spatial Information Still Exists

Even deeper feature maps still have spatial dimensions.

Example:

```text
Input:
224 × 224 × 3

Early Layer:
224 × 224 × 64

Later:
112 × 112 × 128

Later:
56 × 56 × 256

Later:
28 × 28 × 512
```

The spatial resolution becomes smaller.

But each activation sees a larger region of the original image.

So deeper layers often have:

```text
Spatial Resolution ↓
Receptive Field ↑
Representation Complexity ↑
```

---

# ⭐ 12. This Connects Directly to Receptive Field

This is why the previous lecture was important.

Early activation:

```text
small RF
↓
sees local pixels
```

Deep activation:

```text
large RF
↓
sees combinations across a larger part of the image
```

Therefore:

> **increasing receptive field enables deeper layers to integrate broader spatial context**

And this helps build more complex features.

---

# 🧠 13. Local Detail vs Semantic Abstraction

A useful mental model:

### Early Layers

Strong in:

> local detail

Examples:

* exact edges
* fine texture
* local color transitions

### Deep Layers

Stronger in:

> abstract representations

Examples:

* parts
* shape combinations
* category-relevant patterns

Conceptually:

```text
Early CNN
Local + Detailed
↓
Middle CNN
Structured Patterns
↓
Deep CNN
Broader + More Abstract
```

---

# 📉 14. Spatial Resolution Usually Decreases

CNN architectures often include:

* pooling
* stride-2 convolution

These reduce feature-map dimensions.

Example:

```text
224×224
↓
112×112
↓
56×56
↓
28×28
↓
14×14
↓
7×7
```

At the same time, channel depth often increases:

```text
64
↓
128
↓
256
↓
512
```

So a common CNN pattern is:

```text
Height / Width ↓
Number of Channels ↑
```

---

# 🎛️ 15. Why Increase Channels?

As the network goes deeper, it may need to represent:

> more kinds of learned patterns

Early layer:

```text
64 filters
```

Later layer:

```text
256 filters
```

Each filter produces another feature map.

So more channels give the network more capacity to represent:

> different learned feature types

---

# ⚠️ Important

More channels do NOT necessarily mean:

> “more objects”

A channel is simply:

> one learned feature representation

Its meaning depends on what the network learns.

---

# 🧩 16. Multiple Channels Enable Rich Combinations

Suppose a middle layer has feature channels responding to:

```text
Channel 1 → curve-like pattern
Channel 2 → vertical structure
Channel 3 → dark-light transition
Channel 4 → texture pattern
...
```

A deeper filter has weights across:

> all input channels

So it can combine them.

Conceptually:

```text
Curves
+
Edges
+
Texture
+
Shape fragments
↓
More complex representation
```

This is why:

> **multiple filters + channels are fundamental to hierarchical learning**

---

# 🔄 17. A Layer Learns Features From Features

This is perhaps the most important sentence of this lecture:

> **A CNN does not only learn features from pixels; deeper CNN layers learn features from earlier features.**

Layer 1:

`features from pixels`

Layer 2:

`features from Layer 1 features`

Layer 3:

`features from Layer 2 features`

and so on.

Therefore:

```text
Raw Data
↓
Representation
↓
Representation of Representation
↓
Representation of Representation of Representation
```

This creates increasing abstraction.

---

# 🧠 18. Why Nonlinearity Matters Here

Suppose we had:

```text
Conv
↓
Conv
↓
Conv
```

with no nonlinear activation between them.

Then multiple linear convolutional operations could behave like:

> a larger linear transformation

The network would lose much of its ability to build rich nonlinear representations.

Typically we have:

```text
Conv
↓
ReLU / GELU-like activation
↓
Conv
↓
Activation
↓
Conv
```

Nonlinearity allows the network to create:

> increasingly complex feature combinations

This connects directly to our earlier activation-function lectures.

---

# 🔗 19. CNN Feature Hierarchy Connects Many Previous Concepts

Notice how many concepts come together here.

### Convolution

Extracts local patterns.

### Filters

Learn different pattern detectors.

### Feature Maps

Store where and how strongly patterns occur.

### Multiple Channels

Allow many feature types simultaneously.

### Parameter Sharing

Lets the same pattern detector operate across locations.

### Receptive Field

Determines how much context an activation can integrate.

### Depth

Allows features to be composed into more complex features.

### Nonlinearity

Allows nonlinear feature combinations.

So:

> **Feature hierarchy is not a separate trick. It emerges from the CNN architecture we have already built.**

---

# 🧠 20. Hierarchy Example — Car Image

Suppose the input contains a car.

A conceptual progression:

```text
Raw Pixels
↓
Edges
↓
Curves / Corners
↓
Circular Patterns
↓
Wheel-like Structures
↓
Body Shapes
↓
Car-related Representation
```

Again, this is conceptual.

The network is not explicitly programmed with:

```text
if circle → wheel
if wheel + body → car
```

Everything is learned through:

> **loss + backpropagation + optimization**

---

# 🏋️ 21. How Are These Features Learned?

Initially:

> filters contain initialized weights

They do not know:

* edges
* wheels
* faces
* textures

Training proceeds:

```text
Input
↓
Forward Pass
↓
Prediction
↓
Loss
↓
Backpropagation
↓
Update Filters
```

Over many examples:

> filters that help reduce the task loss are reinforced through optimization.

So the hierarchy is:

> **learned automatically from data**

---

# 🧠 22. Task Determines the Feature Hierarchy

Imagine two CNNs.

### CNN A

Trained for:

> dog vs cat classification

### CNN B

Trained for:

> detecting cracks in concrete

Their learned features may differ substantially.

CNN A may need patterns related to:

* fur
* ears
* eyes
* body shapes

CNN B may focus more on:

* fine lines
* discontinuities
* crack textures
* surface patterns

Therefore:

> **feature hierarchy is task-dependent**

---

# 🔬 23. Dataset Also Matters

Suppose you train on:

### Natural Images

The CNN may learn useful patterns involving:

* edges
* textures
* colors
* shapes

### Medical Images

Useful representations might involve:

* tissue boundaries
* anatomical textures
* lesion patterns

### Satellite Images

Features might involve:

* roads
* rooftops
* vegetation patterns
* land structures

So:

> **CNN features are learned representations, not predefined universal labels**

---

# 🖼️ 24. Early Features Often Transfer Better

An interesting practical consequence:

Early features are often relatively general.

For natural-image CNNs, early layers may capture things such as:

* edges
* gradients
* textures

These can be useful across many vision tasks.

Deeper layers tend to become:

> more specialized toward the training task

This is one reason transfer learning often works well.

---

# 🔄 25. Connection to Transfer Learning

Suppose a CNN was trained on a large image dataset.

We want to use it for another image task.

Instead of training from scratch:

```text
Pretrained CNN
↓
Reuse Early / Middle Representations
↓
Adapt Later Layers
```

Why does this work?

Because many low-level visual structures are useful across tasks.

For example:

> edges remain useful whether the final task is cats, cars, or flowers.

We will study transfer learning separately later, but its foundation is:

> **hierarchical feature learning**

---

# 📐 26. Feature Hierarchy vs Feature Map

Do not confuse these.

### Feature Map

A tensor/map produced by a particular filter.

Example:

`56×56`

### Feature Hierarchy

The progression of increasingly complex representations across layers.

Example:

```text
edges
↓
textures
↓
parts
↓
higher-level patterns
```

So:

> **Feature map = representation at a layer/channel**

> **Feature hierarchy = progression across depth**

---

# 🎛️ 27. Feature Hierarchy vs Receptive Field

Also different.

### Receptive Field

Answers:

> How much original input can influence this activation?

### Feature Hierarchy

Answers:

> How representations become progressively more complex across depth.

They are connected because:

> larger receptive fields allow deeper representations to combine information across larger regions.

But they are not the same concept.

---

# 🗺️ 28. What Happens to Location Information?

As we go deeper:

* pooling may reduce resolution
* stride may reduce resolution
* activations represent broader regions

Therefore precise location information may become:

> less detailed

But CNNs do not instantly lose all spatial information.

A deep tensor such as:

`14×14×512`

still contains a:

> spatial grid

Each cell corresponds approximately to a region of the original image.

---

# ⚠️ 29. Why This Matters for Different Tasks

For image classification:

> broad semantic understanding is extremely useful.

For tasks requiring exact location:

* segmentation
* object detection
* keypoint estimation

we also need:

> fine spatial information

This is why many architectures preserve or recover spatial detail using techniques such as:

* skip connections
* multi-scale features
* upsampling

We will encounter these concepts later.

---

# 🧠 30. Classification Example

Imagine:

```text
Input:
Dog Image
```

Early layers may represent:

```text
edges
fur textures
color transitions
```

Middle layers:

```text
curved contours
ear-like patterns
eye-region patterns
```

Deep layers:

```text
combinations strongly associated with dog-like structures
```

Then the final classifier may use those deep representations to produce:

```text
Dog → 0.92
Cat → 0.05
Wolf → 0.03
```

The classifier does not normally operate directly on raw pixels.

It operates on:

> **features learned by the CNN backbone**

---

# 🏗️ 31. CNN as a Feature Extractor

We can conceptually divide a CNN into:

```text
Input Image
      ↓
Convolutional Backbone
      ↓
Hierarchical Feature Extraction
      ↓
Final Representation
      ↓
Classification Head
      ↓
Prediction
```

The convolutional layers act as:

> **learned feature extractors**

This is much more powerful than manually designing features.

---

# 🧠 32. Before Deep Learning — Handcrafted Features

Traditional computer vision often involved humans designing feature extractors.

Examples historically included:

* edge detectors
* corner detectors
* texture descriptors
* SIFT
* HOG

Pipeline:

```text
Image
↓
Human-designed Feature Extraction
↓
Machine Learning Model
```

CNNs changed this approach.

Now:

```text
Image
↓
Learned Feature Extraction
↓
Prediction
```

The feature extractor itself is trained.

---

# ⭐ 33. Why This Was a Major Breakthrough

CNNs can learn:

> **features appropriate for the task automatically**

Instead of requiring engineers to manually decide:

```text
Which edges?
Which shapes?
Which textures?
Which combinations?
```

the network learns useful representations from data.

This ability is called:

> **representation learning**

---

# 🧠 34. CNNs Are Representation-Learning Systems

This is a broader deep-learning principle.

Deep learning is powerful because intermediate layers learn:

> **representations of the data**

CNNs specialize this idea for spatial data.

So the hierarchy:

```text
Pixels
→ Low-Level Features
→ Mid-Level Features
→ Higher-Level Features
```

is an example of:

> **deep representation learning**

---

# 🧩 35. Low-Level vs Mid-Level vs High-Level Features

A useful interview-friendly classification:

| Layer Region | Typical Representation                                    |
| ------------ | --------------------------------------------------------- |
| Early        | edges, gradients, color contrasts, simple textures        |
| Middle       | curves, motifs, textures, shapes, combinations            |
| Deep         | larger structures, parts, task-relevant abstract patterns |

But remember:

> this is a useful intuition, not a strict rule for every CNN.

---

# ⚠️ 36. Do Deeper Layers Always Mean “More Human-Understandable”?

No.

Deeper representations may be:

> highly useful mathematically

without being easy for humans to interpret.

A channel might respond to a complicated combination of:

* texture
* shape
* color
* context

that does not have a simple English name.

So:

> **higher-level does not necessarily mean easily interpretable**

---

# 🔄 37. Distributed Representation

Suppose the network represents the concept of a wheel.

It may not use:

```text
Channel 201 = wheel
```

Instead, information could be distributed across:

```text
Channel 21
Channel 45
Channel 103
Channel 177
...
```

Together their activation pattern may encode useful information.

This is called:

> **distributed representation**

---

# 🧠 38. Why Depth Is Important

Why not learn everything in one convolution layer?

Because a shallow layer would need to directly map:

```text
pixels
↓
complex global patterns
```

Depth allows the problem to be decomposed:

```text
simple
↓
slightly complex
↓
more complex
↓
highly abstract
```

This can make representation learning much more efficient.

---

# 🔗 39. Connection to Universal Approximation

Earlier we learned:

> a sufficiently capable neural network can represent broad classes of functions.

But:

> **Can Represent ≠ Efficient Representation**

CNN depth and hierarchy provide structure.

Instead of learning a giant arbitrary mapping:

```text
pixels → object
```

the CNN builds reusable intermediate representations.

This is one reason architecture matters even though neural networks are theoretically powerful approximators.

---

# 🧱 40. A Full Hierarchical Example

Consider:

```text
224×224×3
```

### Stage 1

```text
Conv
↓
224×224×64
```

Possible representation:

> simple local patterns

---

### Stage 2

```text
Pool / Strided Conv
↓
112×112×128
```

Possible representation:

> combinations of local structures

---

### Stage 3

```text
56×56×256
```

Possible representation:

> larger textures and shapes

---

### Stage 4

```text
28×28×512
```

Possible representation:

> larger structural patterns / parts

---

### Stage 5

```text
7×7×512
```

Possible representation:

> broad task-relevant representations

Then:

```text
Global Pool / Classifier
↓
Prediction
```

---

# 📉 41. The Important Architectural Pattern

As depth increases, we often see:

```text
H × W
↓↓↓

Channels
↑↑↑

Receptive Field
↑↑↑

Feature Abstraction
↑↑↑
```

This is not an absolute law, but it is a very common CNN design pattern.

---

# 🎯 42. Why Reduce Spatial Resolution?

Imagine keeping:

`224×224`

resolution through hundreds of channels and many layers.

That would require huge:

* computation
* memory

Downsampling helps make deeper processing manageable.

At the same time:

> each later activation summarizes information from a broader input region.

So downsampling supports efficient hierarchical processing.

---

# ⚠️ 43. But Downsampling Has a Cost

If you downsample too aggressively:

> small details can disappear.

Suppose a tiny object is only:

`4×4 pixels`

Aggressive stride/pooling early in the network may destroy information needed to detect it.

Therefore CNN architecture design balances:

```text
Efficiency
vs
Spatial Detail
vs
Context
```

---

# 🧠 44. Feature Hierarchy Is Not Manually Programmed

An important distinction:

We design the architecture:

* number of layers
* filter sizes
* channels
* stride
* pooling

But we generally do not manually decide:

```text
Layer 1 = edge detector
Layer 2 = eye detector
Layer 3 = face detector
```

Those representations emerge during:

> **training**

---

# 🔄 45. End-to-End Learning

Because the entire CNN is trained together:

```text
Image
↓
Early Features
↓
Middle Features
↓
Deep Features
↓
Prediction
↓
Loss
```

the loss gradient flows backward through:

> all layers

Therefore early filters are also optimized based on their contribution to the final task.

This is:

> **end-to-end feature learning**

---

# 🎯 46. A Crucial Insight

An edge detector is not useful merely because:

> edges are interesting.

It becomes useful because:

> detecting that pattern helps downstream layers reduce final prediction loss.

That is the difference between:

> handcrafted feature engineering

and:

> learned representation hierarchy.

---

# 🧠 47. What Happens During Backpropagation?

Suppose a deeper layer needs a certain pattern to classify cats correctly.

Gradients can influence earlier filters so that they produce:

> useful intermediate representations

for later layers.

So the hierarchy is not learned independently layer by layer.

In modern end-to-end CNN training:

> **all layers cooperate through the optimization objective**

---

# 📍 48. Hierarchy Exists Across Both Depth and Channels

There are two dimensions to remember.

### Across Channels

At one layer:

> multiple filters learn different feature representations.

### Across Depth

Later layers:

> combine earlier feature representations into increasingly complex ones.

So:

```text
Width / Channels
→ variety of features

Depth
→ hierarchy of features
```

---

# 🎨 49. A Better Mental Visualization

Imagine a factory.

### Stage 1

Receives raw material:

> pixels

### Stage 2

Produces basic components:

> edges / textures

### Stage 3

Combines components:

> curves / shapes

### Stage 4

Builds larger assemblies:

> parts

### Stage 5

Produces final representation:

> task-relevant semantic structure

Each stage receives:

> the output of the previous stage

not the original raw material alone.

---

# 🧠 50. Feature Hierarchy in One Diagram

```text
             INPUT IMAGE
                 │
                 ▼
          Raw RGB Pixels
                 │
                 ▼
        ┌─────────────────┐
        │ Early CNN Layer │
        └─────────────────┘
                 │
        Edges / Gradients
        Colors / Textures
                 │
                 ▼
        ┌─────────────────┐
        │ Middle CNN      │
        └─────────────────┘
                 │
       Curves / Shapes /
       Pattern Combinations
                 │
                 ▼
        ┌─────────────────┐
        │ Deeper CNN      │
        └─────────────────┘
                 │
       Broader / Task-Relevant
       Representations
                 │
                 ▼
             CLASSIFIER
                 │
                 ▼
             PREDICTION
```

---

# ⚠️ 51. Common Misconceptions

### ❌ “Every first-layer filter detects an edge.”

Not guaranteed.

Early filters often learn simple local patterns, but exact behavior depends on training.

---

### ❌ “Every deeper neuron corresponds to an object part.”

Not necessarily.

Representations may be distributed and difficult to interpret.

---

### ❌ “Feature hierarchy is manually programmed.”

No.

The architecture is designed, but representations are learned.

---

### ❌ “Deep layers no longer have spatial information.”

Not necessarily.

Many deep feature maps still have H×W structure.

---

### ❌ “More channels means more classes.”

No.

Channels represent learned feature dimensions, not class count.

---

### ❌ “Bigger receptive field automatically creates semantic understanding.”

No.

It provides access to broader context.

Useful representations still need to be learned.

---

# 🎤 52. Interview Question — Why Do CNN Features Become More Complex With Depth?

A strong answer:

> **Each convolutional layer receives feature maps from the previous layer rather than operating independently on raw pixels. Early layers usually capture simple local patterns because they have small receptive fields. Deeper layers combine these learned features across channels and over increasingly larger receptive fields, allowing the network to construct progressively more complex and task-relevant representations. Nonlinear activations between layers make these compositions more expressive.**

---

# 🎤 53. Interview Question — Why Are Early CNN Features Often More Generic?

Because early layers primarily operate on:

> local visual structures

such as:

* edges
* gradients
* color changes
* simple textures

These patterns occur in many visual tasks.

Deeper layers depend more heavily on:

> task-specific combinations of those patterns.

Therefore:

> **earlier representations often transfer more easily between related image tasks.**

---

# 🎤 54. Interview Question — What Is the Relationship Between Receptive Field and Feature Hierarchy?

A strong answer:

> **Receptive field determines how much of the original input can influence an activation. Early layers have small receptive fields and therefore mainly model local structures. As depth, stride, pooling, or dilation increase the receptive field, deeper activations can combine information from broader regions, supporting more complex hierarchical representations. However, a larger receptive field alone does not guarantee semantic features—the representations must still be learned during training.**

---

# ⚡ 55. Quick Recall

| Concept                 | Meaning                                  |
| ----------------------- | ---------------------------------------- |
| Feature                 | Useful learned input pattern             |
| Feature Hierarchy       | Increasing abstraction across depth      |
| Early Features          | Often local/simple patterns              |
| Middle Features         | Pattern combinations / shapes / textures |
| Deep Features           | Broader task-relevant representations    |
| Channels                | Different learned feature dimensions     |
| Depth                   | Enables composition of features          |
| Receptive Field         | Context available to an activation       |
| Downsampling            | Resolution ↓, usually RF growth ↑        |
| Nonlinearity            | Enables richer feature composition       |
| Representation Learning | Learning features automatically          |
| Transfer Learning       | Reusing learned representations          |

---

# 🧠 56. Mental Model

```text
CNN does NOT learn:

Pixels
──────────────→ Object
      one jump
```

Instead:

```text
Pixels
  ↓
Simple Features
  ↓
Feature Combinations
  ↓
Larger Structures
  ↓
Task-Relevant Representation
  ↓
Prediction
```

That gradual transformation is the:

> **CNN Feature Hierarchy**

---

# ⭐ Golden Rule

> **CNN depth turns local patterns into increasingly rich representations: early layers learn features from pixels, while deeper layers learn features from features.**
