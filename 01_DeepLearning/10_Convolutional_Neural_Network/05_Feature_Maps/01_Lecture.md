# 📘 Lecture — Feature Maps in CNNs

> **Core Idea:** A feature map is the spatial output produced by one filter. It tells us **where** that filter responded and **how strongly** it responded across the input.

---

# 🎯 Learning Objectives

By the end of this lecture, you should understand:

- 🗺️ What a feature map actually is
- 🎛️ Why one filter produces one feature map
- 📍 How spatial information is preserved
- 📈 What individual activation values mean
- 📦 Why multiple filters create multiple output channels
- 🔄 How feature maps become input to the next CNN layer
- 🧠 Why deeper feature maps represent learned features rather than RGB colors
- 🏗️ How feature maps become more abstract in deeper layers
- ⚠️ Why feature map values are activations, not trainable parameters

---

# 🗺️ 1. From Filter to Feature Map

In the previous lecture, we learned:

> A filter is a small trainable weight tensor.

Now suppose one filter scans across the input.

At every location it produces:

> **One scalar activation**

Example:

Position 1 → `2`  
Position 2 → `0`  
Position 3 → `5`

Position 4 → `1`  
Position 5 → `8`  
Position 6 → `2`

Position 7 → `0`  
Position 8 → `3`  
Position 9 → `1`

Arrange those outputs spatially:

2   0   5  
1   8   2  
0   3   1

This complete grid is:

> **A Feature Map**

---

# 🎛️ 2. One Filter → One Feature Map

This relationship is fundamental.

Suppose we have:

Filter A

It scans the input and produces:

Feature Map A

Then:

Filter B  
→ Feature Map B

Filter C  
→ Feature Map C

So:

> **1 filter → 1 feature map**

If a convolutional layer has:

`32 filters`

then it produces:

`32 feature maps`

---

# 📈 3. What Does One Feature-Map Value Mean?

Suppose a feature map contains:

0   1   2  
1   9   0  
2   1   0

The value:

`9`

means:

> The filter produced a strong response at that spatial location.

The value itself is:

> **an activation**

It is not:

- a filter weight
- a pixel value from the original image
- a new trainable parameter

It is simply the result of applying the learned filter to that local input region.

---

# 📍 4. Feature Maps Preserve Spatial Information

This is one of the most important ideas in CNNs.

Suppose a filter detects a strong pattern near the center of the image.

Then its feature map may look like:

0   0   1  
0   8   1  
0   1   0

The high activation near the center tells us:

> the filter responded strongly near the corresponding region of the input.

So a feature map stores two things:

- 📈 **response strength**
- 📍 **spatial location**

That is why we call it a:

> **feature map**

It maps a learned feature response across space.

---

# ⚠️ 5. Feature Map Does NOT Mean Exact Original Pixel Coordinates

The feature map preserves spatial structure, but not always one-to-one with the original image.

For example:

Input:

`32 × 32`

Feature Map:

`30 × 30`

or:

`16 × 16`

depending on operations such as:

- Kernel size
- Stride
- Padding
- Pooling

So when we say spatial information is preserved, we mean:

> **relative spatial structure is preserved**

not necessarily:

> exact original pixel coordinate = exact feature-map coordinate

We will study these dimension changes in later lectures.

---

# 🔍 6. Think of a Feature Map as a Response Map

A very useful mental model is:

> **A feature map shows where a filter responded strongly or weakly.**

Imagine a filter that has learned to respond to some local pattern.

Then:

Low Activation  
→ weak response

High Activation  
→ strong response

So the feature map acts like a:

> **response map for that learned filter**

---

# 🧠 7. A Feature Map Is NOT Necessarily Human-Interpretable

For intuition, we often say:

Filter 1  
→ vertical edge

Filter 2  
→ horizontal edge

Filter 3  
→ texture

Then their feature maps seem easy to understand.

But in a real trained CNN:

> Not every feature map has a simple human-readable meaning.

A learned filter might respond to a complex combination of:

- intensity changes
- colors
- textures
- shapes
- previous learned features

So:

> **Feature maps are learned representations, not manually labeled semantic maps.**

---

# 📦 8. Multiple Feature Maps Form the Output Tensor

Suppose a convolutional layer uses:

`16 filters`

Each filter produces:

`1 feature map`

So the layer produces:

`16 feature maps`

These are stacked together.

Conceptually:

Feature Map 1  
Feature Map 2  
Feature Map 3  
...  
Feature Map 16

Together they form the output representation.

If using:

`H × W × C`

notation:

`H_out × W_out × 16`

So:

> **Number of feature maps = Number of output channels**

---

# 🌈 9. Why Are Feature Maps Called Channels?

Suppose the original input is RGB:

Input Channels:

- Red
- Green
- Blue

So:

`C_in = 3`

After convolution with 32 filters:

Output Channels:

- Feature Map 1
- Feature Map 2
- ...
- Feature Map 32

So:

`C_out = 32`

These 32 channels are no longer:

Red  
Green  
Blue

They are:

> **learned feature channels**

---

# 🔄 10. Feature Maps Become Input to the Next Layer

This is where CNNs become powerful.

Suppose:

Layer 1  
→ produces 32 feature maps

Then Layer 2 receives:

> **those 32 feature maps as its input channels**

So:

Input Image  
↓  
Conv Layer 1  
↓  
32 Feature Maps  
↓  
Conv Layer 2  
↓  
64 New Feature Maps  
↓  
Conv Layer 3  
↓  
More Advanced Features

This allows the network to build increasingly complex representations.

---

# 🧠 11. What Does the Next Filter Actually See?

Suppose Layer 1 outputs:

Feature Map 1  
Feature Map 2  
Feature Map 3  
...  
Feature Map 32

A filter in Layer 2 spans:

> all 32 input feature maps

It therefore sees combinations of previously learned responses.

That allows Layer 2 to learn patterns like:

> "When feature A and feature B occur together in this local region, produce a strong response."

This is how simple learned features combine into more complex ones.

---

# 🏗️ 12. Feature Maps Become More Abstract with Depth

Early layers may produce feature maps responding to:

- edges
- simple textures
- color transitions

Middle layers may respond to combinations such as:

- corners
- curves
- shapes
- texture combinations

Deeper layers may respond to:

- object parts
- complex structures
- high-level representations

Conceptually:

Raw Pixels  
↓  
Simple Feature Maps  
↓  
Intermediate Feature Maps  
↓  
Complex Feature Maps  
↓  
High-Level Representation

---

# 🧩 13. Why Do We Need Many Feature Maps?

Imagine trying to represent an image using only one learned feature.

That would be extremely limiting.

Instead, multiple feature maps allow the network to represent:

> many different learned properties at the same time

For example:

Feature Map 1  
→ responds strongly in one type of region

Feature Map 2  
→ responds to another pattern

Feature Map 3  
→ responds to another combination

Together:

> they form a richer representation of the input.

---

# 🔁 14. Feature Maps Change for Different Inputs

Important distinction:

### Filter weights

Stay fixed during one forward pass.

### Feature-map values

Depend on the input.

So if the same filter sees:

Image A  
→ Feature Map A

Image B  
→ Feature Map B

The filter is the same.

But the feature maps differ because:

> the input content is different.

---

# 🎛️ 15. Filter vs Feature Map

This distinction must be crystal clear.

### 🎛️ Filter

Learned parameters

Example:

1   0  -1  
1   0  -1  
1   0  -1

### 🗺️ Feature Map

Computed activations

Example:

0   3   7  
1   0   4  
2   6   1

So:

Filter  
→ **what the model learns**

Feature Map  
→ **what the filter produces for a specific input**

---

# 🧮 16. One Activation vs Full Feature Map

Do not confuse:

### One convolution position

Produces:

`1 scalar`

### Same filter across all positions

Produces:

`1 feature map`

### Multiple filters

Produce:

`multiple feature maps`

So:

One Position  
→ One Activation

One Filter  
→ One Feature Map

N Filters  
→ N Feature Maps

---

# 🔥 17. Feature Maps Before and After Activation Functions

Suppose convolution produces:

-4   2   5  
 1  -3   6

This is the convolution output.

If ReLU follows:

`ReLU(x) = max(0, x)`

then:

0   2   5  
1   0   6

Depending on context, people may call either of these:

> **feature maps**

So you may encounter:

Pre-activation feature map  
→ before ReLU

Post-activation feature map  
→ after ReLU

For most high-level discussions, the exact naming convention is less important than understanding:

> convolution produces spatial activations, and nonlinearities may transform them afterward.

---

# ⚠️ 18. High Activation Does NOT Mean "Feature Definitely Exists"

Suppose a feature map contains a high value.

It is tempting to say:

> "The feature is definitely present."

That is too strong.

A high activation means:

> **the learned filter responded strongly to that local input**

This does not automatically mean:

- the filter has a perfect semantic interpretation
- the model is certain about an object
- one activation alone determines the prediction

CNN decisions are usually based on:

> many feature maps and many layers working together.

---

# 📊 19. Feature Maps Are Input-Dependent Activations

This distinction is useful for interviews.

### Trainable quantities

- Filter weights
- Biases

### Computed quantities

- Feature maps
- Activations

During forward propagation:

Input  
+  
Learned Filter  
↓  
Feature Map

During backpropagation:

Loss  
↓  
Gradients  
↓  
Filter weights updated

The feature map itself is not usually stored as a persistent trainable parameter.

---

# 🎯 20. Complete Example

Suppose:

Input:

`32 × 32 × 3`

Conv Layer:

- `16 filters`
- each filter spans all 3 input channels

Then:

Each Filter  
↓  
Produces One Feature Map

So:

16 Filters  
↓  
16 Feature Maps

Output:

`H_out × W_out × 16`

Those 16 feature maps then become:

> **16 input channels for the next convolutional layer**

That next layer may use:

`32 filters`

and therefore produce:

`32 new feature maps`

---

# 🧠 21. The Representation Story

This is the key CNN story:

Original Image  
↓  
Raw Pixel Channels  
↓  
Convolution  
↓  
Feature Maps  
↓  
Next Convolution  
↓  
New Feature Maps  
↓  
More Abstract Representation  
↓  
Eventually Useful for Prediction

The network gradually transforms:

> **raw data into learned representations**

Feature maps are the intermediate representations that make this possible.

---

# ⚡ Quick Recall Table

| Concept | Meaning |
|---|---|
| Feature Map | Spatial activations from one filter |
| One Position | One activation |
| One Filter | One feature map |
| N Filters | N feature maps |
| Feature-Map Value | Strength of filter response |
| Feature-Map Position | Approximate spatial location |
| Output Channels | Number of feature maps |
| Next Layer Input | Previous layer's feature maps |
| Early Feature Maps | Often simpler patterns |
| Deep Feature Maps | More abstract learned representations |

---

# 🧠 Mental Model

🎛️ One Learned Filter  
↓  
Scans Local Regions  
↓  
Produces One Activation at Each Position  
↓  
Activations Arranged Spatially  
↓  
🗺️ One Feature Map

Multiple Filters  
↓  
Multiple Feature Maps  
↓  
Stacked as Output Channels  
↓  
Become Input Channels for Next Layer

---

# 🎤 30-Second Interview Answer

> **A feature map is the spatial activation output produced by one convolutional filter. Each value shows how strongly the filter responded to a local region, while the spatial arrangement preserves where those responses occurred. One filter produces one feature map, so multiple filters produce multiple output channels. These feature maps then become the input channels to the next convolutional layer, allowing deeper layers to learn increasingly abstract representations.**

---

# 📌 Key Takeaways

- 🗺️ Feature maps are activation maps, not trainable parameters
- 🎛️ One filter produces one feature map
- 📈 Each value represents filter-response strength
- 📍 Spatial arrangement preserves relative location
- 📦 Multiple feature maps form output channels
- 🔄 Output feature maps become input channels for the next layer
- 🌈 Deeper channels are learned features, not RGB colors
- 🏗️ Feature maps usually become more abstract with depth
- ⚠️ High activation means strong response, not absolute semantic certainty
- 🔥 Feature maps may refer to pre- or post-activation outputs depending on convention

---

# ⭐ Golden Rule

> **A feature map is a spatial response map: it shows where one learned filter responded and how strongly it responded across the input.**

---

# ➡️ Next Topic

## 📘 Stride

Next we will study:

- ➡️ How far the filter moves at each step
- 📏 What stride actually controls
- 🗺️ Why larger stride reduces feature-map size
- 🧮 How stride affects output dimensions
- ⚖️ Trade-offs between spatial detail and computation