# 📘 Lecture — Parameter Sharing in CNNs

> **Core Idea:** A convolutional filter does not learn a new set of weights for every image location. The **same filter weights are reused across all spatial positions**. This is called **parameter sharing**, and it is one of the main reasons CNNs require far fewer parameters than fully connected networks.

---

# 🎯 Learning Objectives

By the end of this lecture, you should understand:

* 🔁 What parameter sharing means
* 🧠 Why CNNs reuse the same filter across spatial locations
* 🗺️ How one filter detects the same pattern at different locations
* 🧮 How parameter sharing reduces parameter count
* 🏗️ Why fully connected layers do not have the same spatial sharing
* 📍 The connection between parameter sharing and translation equivariance
* 🎛️ How sharing works with multiple filters and channels
* ⚖️ Benefits and limitations of parameter sharing
* 🚫 Common misconceptions around shared filters

---

# 🧠 1. Start With a Simple Question

Suppose we want a CNN to detect:

> **a vertical edge**

in an image.

The vertical edge might appear:

* on the left
* in the center
* on the right
* near the top
* near the bottom

Should the network learn:

> one vertical-edge detector for the top-left corner,

another one for:

> the center,

and another for:

> the bottom-right?

That would be extremely wasteful.

Instead, CNNs use a powerful assumption:

> **If a pattern is useful in one location, the same pattern may also be useful somewhere else.**

This motivates:

# 🔁 Parameter Sharing

---

# 🎛️ 2. Recall What a Filter Does

Suppose we have a:

`3 × 3`

filter.

Conceptually:

```text
w1  w2  w3
w4  w5  w6
w7  w8  w9
```

The filter is placed over one local region of the image.

It computes:

```text
Local Patch
×
Filter Weights
↓
Sum
↓
One Response
```

Then the filter moves to the next position.

---

# 🔁 3. What Happens When the Filter Moves?

Here is the key point.

When the filter moves:

> **we do NOT create a new set of weights**

The same:

```text
w1  w2  w3
w4  w5  w6
w7  w8  w9
```

is used again.

So:

```text
Position 1 → same filter
Position 2 → same filter
Position 3 → same filter
...
Position N → same filter
```

That is:

> **Parameter Sharing**

---

# 🗺️ 4. The Filter Moves, the Parameters Do Not Change

Imagine an image:

```text
+----------------------+
|                      |
|   Filter Position 1  |
|                      |
|           Position 2 |
|                      |
| Position 3           |
+----------------------+
```

At every position:

> the exact same learned filter weights are applied.

So if a filter has:

`9 weights`

it still has:

> **9 weights**

regardless of whether it is used at:

* 10 positions
* 100 positions
* 10,000 positions

---

# 🧠 5. Why Is Sharing Reasonable for Images?

Images contain patterns that may appear in many locations.

Examples:

* edges
* corners
* textures
* curves
* object parts

An edge does not become a completely different visual concept simply because it moved:

`20 pixels to the right`

Therefore CNNs encode the prior:

> **Useful local patterns can occur at different spatial locations.**

Instead of learning separate detectors everywhere:

> one detector is reused everywhere.

---

# 🔍 6. Example — Detecting the Same Edge in Different Places

Suppose a filter learns to respond strongly to:

> vertical-edge-like patterns

Now imagine the edge appears on the left side.

The filter scans there:

```text
Edge
↓
Strong Response
```

Now the same edge appears on the right side.

The same filter reaches that location:

```text
Same Edge
↓
Same Learned Filter
↓
Strong Response
```

No new edge detector is required.

---

# 🗺️ 7. Feature Maps Are Possible Because of Sharing

Recall:

> One filter scanning across an input produces one feature map.

Why?

Because the same detector is evaluated at:

> many spatial positions.

So a feature map tells us:

> **where and how strongly this same learned pattern responds across the input**

Conceptually:

```text
One Shared Filter
↓
Position 1 → response
Position 2 → response
Position 3 → response
...
↓
Feature Map
```

---

# 🧮 8. What If We Did NOT Share Parameters?

Suppose input is:

`32 × 32`

and we use a:

`3 × 3`

local detector.

With:

* no padding
* stride 1

output becomes:

`30 × 30`

So there are:

`900 output positions`

Now imagine every position had its own independent:

`3 × 3`

weights.

Weights would be:

`900 × 9`

= `8,100`

weights.

For just:

> **one output feature type**

---

# 🔁 9. With Parameter Sharing

Instead, CNN uses:

> one shared `3 × 3` filter

Weights:

`3 × 3`

= `9`

That same set of:

`9 weights`

is reused across all:

`900 positions`.

So:

Without Sharing:

`8,100 weights`

With Sharing:

> **9 weights**

That is a dramatic reduction.

---

# 🌈 10. Multi-Channel Case

Now suppose input is RGB:

`32 × 32 × 3`

One `3×3` filter has shape:

`3 × 3 × 3`

Weights:

`27`

The full:

`3 × 3 × 3`

filter is shared across spatial positions.

Important:

> Parameter sharing applies to the complete multi-channel filter.

It is not just one `3×3` slice being reused.

---

# 🎛️ 11. What If We Have Multiple Filters?

Suppose:

`C_in = 3`

Kernel:

`3 × 3`

Filters:

`64`

Each filter has:

`3 × 3 × 3`

weights.

So:

Filter 1
→ shared spatially

Filter 2
→ shared spatially

...

Filter 64
→ shared spatially

Different filters have:

> **different learned parameters**

but each individual filter is reused across H and W.

---

# ⚠️ 12. Parameter Sharing Does NOT Mean All Filters Have the Same Weights

This is a very common misunderstanding.

Wrong:

> "CNN parameter sharing means every filter uses the same weights."

No.

Suppose there are:

`64 filters`

Then:

```text
Filter 1 ≠ Filter 2 ≠ Filter 3 ...
```

They learn different patterns.

Sharing means:

> **Filter 1 uses its own weights at every spatial location**

and:

> **Filter 2 uses its own different weights at every spatial location**

---

# 🧠 13. Sharing Is Across Spatial Positions

This is the cleanest rule:

> **CNN parameter sharing happens mainly across spatial locations.**

If one filter has weights:

`W`

then:

```text
Location 1 → W
Location 2 → W
Location 3 → W
...
```

not:

```text
Location 1 → W1
Location 2 → W2
Location 3 → W3
```

---

# 🚫 14. Sharing Is Not the Same as Sharing Across Channels

Remember our previous lecture.

For input with:

`C_in = 64`

one standard filter has:

`3 × 3 × 64`

weights.

Different channel slices can contain:

> different learned values.

So parameter sharing does NOT mean:

> use the identical `3×3` weights for every input channel.

Rather:

> **the entire `3×3×64` filter is reused across spatial positions.**

---

# 🧮 15. Parameter Count Does NOT Depend on H and W

This is one of the most important consequences.

For a standard convolution:

`Parameters = K_h × K_w × C_in × C_out + C_out`

if biases are used.

Notice what is missing:

* `H`
* `W`

Input spatial size does not directly enter the parameter-count formula.

Why?

Because the same filter weights are reused over all H/W locations.

---

# 📐 16. Example — Same Conv, Different Image Sizes

Suppose a convolution uses:

* Kernel = `3 × 3`
* Input channels = `3`
* Output channels = `64`

Parameters:

`3 × 3 × 3 × 64 + 64`

= `1,792`

Now apply the same convolution to:

`32 × 32 × 3`

Parameters:

> `1,792`

Apply it to:

`224 × 224 × 3`

Parameters:

> `1,792`

Apply it to:

`512 × 512 × 3`

Parameters:

> `1,792`

The number of applications changes.

The number of learned parameters does not.

---

# ⚡ 17. Parameters vs Computation

Be careful.

Larger input size does not necessarily increase the number of convolution parameters.

But it does increase:

> **how many times those parameters are used**

So larger H/W usually means:

* more convolution operations
* more activations
* more compute
* more memory

But not necessarily:

> more learned filter parameters.

This distinction is important in interviews.

---

# 🏗️ 18. Compare With a Fully Connected Layer

Suppose input image is:

`224 × 224 × 3`

Flattened size:

`150,528`

Now connect it to:

`1,000 neurons`

Weights:

`150,528 × 1,000`

= `150,528,000`

weights

before biases.

Each spatial input position effectively gets its own learned connection.

There is no CNN-style spatial reuse.

---

# 🎛️ 19. Compare With Convolution

Suppose instead we use:

`64`

filters of size:

`3 × 3`

on RGB input.

Parameters:

`3 × 3 × 3 × 64`

= `1,728 weights`

plus:

`64 biases`

Total:

> **1,792 parameters**

Compare:

Fully Connected:

> ~150 million weights

Convolution:

> 1,792 parameters

These layers do different jobs, so this is not an apples-to-apples architectural replacement.

But the comparison demonstrates why:

> **local connectivity + parameter sharing**

can drastically reduce parameter count.

---

# 🧠 20. Two Important CNN Ideas Work Together

CNN efficiency comes from two closely related ideas:

## 📍 Local Connectivity

A neuron only looks at:

> a local region

instead of the entire image.

---

## 🔁 Parameter Sharing

The same local detector is:

> reused at many spatial positions.

Together:

```text
Local Receptive Region
+
Shared Weights
↓
Far Fewer Parameters
```

---

# 📍 21. Parameter Sharing and Translation

Suppose an object pattern moves to another location.

Because the same filter exists everywhere spatially:

> the CNN can still respond to that pattern.

This gives convolution an important structural property:

> **Translation Equivariance**

---

# 🧠 22. What Is Translation Equivariance?

Very roughly:

If the input pattern shifts:

> the corresponding feature-map response tends to shift as well.

Conceptually:

```text
Input Pattern Moves Right
↓
Feature Response Moves Right
```

The feature does not necessarily disappear just because its location changed.

---

# ⚠️ 23. Equivariance Is NOT Invariance

These words are often confused.

### Translation Equivariance

Input shifts
↓
Output representation shifts correspondingly

---

### Translation Invariance

Input shifts
↓
Output remains unchanged

Convolution itself is more naturally associated with:

> **translation equivariance**

not perfect translation invariance.

---

# 🗺️ 24. Simple Equivariance Intuition

Imagine a vertical line appears at:

`x = 10`

A vertical-edge filter produces a strong response around:

`x = 10`

Now shift the line to:

`x = 20`

Since the same filter scans everywhere:

> the strong response also moves toward `x = 20`.

So:

```text
Input Feature Shift
↓
Feature Map Response Shift
```

This is the intuition behind equivariance.

---

# ⚠️ 25. Is CNN Convolution Perfectly Translation-Equivariant?

In idealized convolution with appropriate assumptions:

> convolution has translation-equivariant structure.

But practical networks can deviate because of:

* stride
* pooling
* padding/boundaries
* discrete sampling
* nonlinear architecture components

So avoid saying:

> "Every CNN is perfectly translation-equivariant under every transformation."

The architectural bias comes from shared convolution filters.

---

# 🧠 26. Why Parameter Sharing Is an Inductive Bias

An inductive bias means:

> an assumption built into the model architecture.

CNNs assume:

> patterns that matter in one region may also matter elsewhere.

This assumption is extremely useful for many visual tasks.

It allows learning from data more efficiently than treating every location as completely unrelated.

---

# 🎯 27. Why Parameter Sharing Improves Data Efficiency

Without sharing:

The model may need to separately learn:

> edge-at-top-left

and:

> edge-at-bottom-right

With sharing:

Learning a useful edge detector from examples in one region allows the same detector to operate:

> everywhere.

This can reduce the amount of independent evidence needed for each location.

---

# ⚖️ 28. But Parameter Sharing Is Also a Restriction

Parameter sharing is powerful because it makes an assumption.

But every assumption can also be limiting.

It assumes:

> the same transformation should be useful across different spatial positions.

What if location itself strongly changes meaning?

For example, in some specialized tasks:

* top of image may systematically differ from bottom
* sensor geometry may be location-dependent
* different spatial regions may follow different statistics

Then strict spatial sharing may not always be optimal.

---

# 🧠 29. Shared Parameters vs Shared Activations

Another important distinction:

Parameters:

> the learned filter weights

Activations:

> the outputs produced when those weights see different input regions

Parameters are shared.

Activations are:

> **not necessarily the same**

Example:

Same Filter
+
Different Image Patch
↓
Different Response

So:

> **same weights ≠ same outputs**

---

# 🎛️ 30. Same Filter, Different Locations, Different Responses

Suppose filter:

`F`

is applied at three positions.

Position 1 contains an edge:

`F → 8.2`

Position 2 contains flat region:

`F → 0.1`

Position 3 contains opposite pattern:

`F → -4.5`

Same parameters.

Different local inputs.

Therefore:

> different activations.

---

# 🧮 31. Full Multi-Filter Example

Input:

`32 × 32 × 3`

Conv:

* `3×3` kernel
* 64 filters
* padding 1
* stride 1

Output:

`32 × 32 × 64`

Each filter contains:

`3 × 3 × 3 = 27 weights`

Those same 27 weights are reused across:

`32 × 32`

spatial positions for that filter.

There are:

`64`

different filters.

Total weights:

`27 × 64`

= `1,728`

not:

`27 × 64 × 32 × 32`

because:

> spatial positions reuse the parameters.

---

# ⭐ 32. Critical Interview Calculation

A developer calculates convolution weights as:

`K_h × K_w × C_in × C_out × H_out × W_out`

Is this correct?

> ❌ No.

`H_out × W_out`

represents:

> how many times the filters are applied

not:

> how many independent filter parameters exist.

Correct:

`Weights = K_h × K_w × C_in × C_out`

---

# 🧠 33. Parameters vs Activations Example

For:

Input:

`32 × 32 × 3`

64 filters:

`3 × 3`

Output:

`32 × 32 × 64`

### Parameters

`3 × 3 × 3 × 64`

= `1,728 weights`

### Output Activations

`32 × 32 × 64`

= `65,536 activations`

These numbers describe completely different things.

---

# 🔁 34. Sharing During Training

During the forward pass:

the same filter is used at all positions.

During backpropagation:

each location where the filter was used can contribute gradient information for:

> **that same shared parameter set**

Conceptually:

```text
Location 1 gradient
+
Location 2 gradient
+
Location 3 gradient
+
...
↓
Gradient for Shared Filter
```

Then the optimizer updates:

> **one shared filter**

not separate copies for every location.

---

# 🧠 35. Why Gradients Accumulate

Suppose weight:

`w1`

inside a filter is used at:

`100 spatial locations`

The final loss may depend on all those uses.

Therefore the gradient:

`dL/dw1`

contains contributions from:

> every location where `w1` participated.

This is how one shared detector learns from patterns occurring throughout the image.

---

# 🔄 36. Weight Update Still Happens Once Per Parameter

Even though a filter may be used thousands of times:

there is still only:

> **one stored parameter value**

for each filter weight.

Backpropagation accumulates its gradient.

Then the optimizer performs an update like:

`w := w - learning_rate × gradient`

on that shared weight.

---

# 🏗️ 37. Parameter Sharing Across a CNN

Suppose:

### Conv 1

`3 × 3 × 3 × 64`

### Conv 2

`3 × 3 × 64 × 128`

### Conv 3

`3 × 3 × 128 × 256`

Within each layer:

> each filter is shared spatially.

But parameters are not usually shared automatically:

> **between different convolutional layers**

Conv 1 and Conv 2 normally have different learned weight tensors.

---

# ⚠️ 38. Sharing Across Space ≠ Sharing Across Layers

Wrong:

> "Because CNNs use parameter sharing, Conv1 and Conv2 use the same filters."

Correct:

Parameter sharing normally refers to:

> reuse of a filter across spatial positions within a convolution operation.

Different layers generally learn:

> different filters.

---

# 🧠 39. Why Deeper Layers Need Different Filters

Early layers may learn patterns such as:

* edge-like structures
* simple textures

Later layers receive learned feature maps and may learn:

* combinations of textures
* parts
* higher-level structures

So forcing every layer to have the same filters would defeat the purpose of:

> hierarchical representation learning.

---

# 📊 40. Fully Connected vs Convolution

| Property                                      | Fully Connected | Convolution               |
| --------------------------------------------- | --------------- | ------------------------- |
| Local connectivity                            | ❌ Usually No    | ✅ Yes                     |
| Spatial parameter sharing                     | ❌ No            | ✅ Yes                     |
| Same detector reused spatially                | ❌               | ✅                         |
| Parameter count depends strongly on input H/W | ✅               | ❌ Usually not directly    |
| Translation-equivariant bias                  | ❌ Not inherent  | ✅ Convolution provides it |

---

# ⚡ 41. Main Benefits of Parameter Sharing

Parameter sharing provides:

### 🧮 Fewer Parameters

Same filter reused across many positions.

### 📊 Better Parameter Efficiency

One learned pattern detector serves many locations.

### 🗺️ Spatial Generalization

A pattern learned in one location can be detected elsewhere.

### 🧠 Useful Inductive Bias

Encodes the assumption that local patterns may repeat spatially.

### 📍 Translation-Equivariant Structure

Shifting the input tends to shift convolutional responses.

---

# ⚠️ 42. Parameter Sharing Does Not Mean Position Is Lost

A common misunderstanding:

> "If the same filter is used everywhere, the CNN doesn't know where a feature occurred."

Wrong.

The feature map still has spatial coordinates.

If a feature occurs in the upper-left:

> activation appears near the upper-left of the feature map.

If it occurs near the bottom-right:

> activation appears near the bottom-right.

So sharing the detector does not automatically erase:

> spatial location.

---

# 🗺️ 43. Same Detector, Spatially Organized Responses

Think of a feature map as:

> a map showing where the shared detector responded.

So parameter sharing gives:

Same Detector Everywhere

while the feature map preserves:

> **where responses occurred**

subject to changes caused by stride, padding, pooling, etc.

---

# 🧩 44. Full Mental Model

```text
Learn One Filter
↓
Reuse Same Filter at Position 1
↓
Reuse Same Filter at Position 2
↓
Reuse Same Filter at Position 3
↓
...
↓
Collect Responses
↓
Feature Map
```

With multiple filters:

```text
Filter 1 shared across space → Feature Map 1
Filter 2 shared across space → Feature Map 2
...
Filter C_out shared across space → Feature Map C_out
```

---

# 🧮 45. Parameter Formula

For standard convolution:

`Weights = K_h × K_w × C_in × C_out`

If biases are used:

`Total Parameters = K_h × K_w × C_in × C_out + C_out`

Notice:

> **No H_out or W_out**

because spatial positions share the same weights.

---

# ⚡ Quick Recall Table

| Concept                  | Key Idea                                      |
| ------------------------ | --------------------------------------------- |
| Parameter Sharing        | Same filter reused across spatial positions   |
| Shared Unit              | Complete `K_h × K_w × C_in` filter            |
| Different Filters        | Have different parameters                     |
| Channels                 | Can have different filter slices              |
| H/W                      | Do not directly increase conv parameter count |
| Larger H/W               | Increase computation, not filter count        |
| Feature Map              | Responses of one shared filter across space   |
| Main Benefit             | Parameter efficiency                          |
| Spatial Generalization   | Same feature detectable in many locations     |
| Translation Equivariance | Shift input → response tends to shift         |
| Invariance               | Not the same as equivariance                  |
| Between Layers           | Parameters usually not automatically shared   |

---

# 🎤 46. 30-Second Interview Answer

> **Parameter sharing in CNNs means that the same convolutional filter weights are reused across all spatial locations. Instead of learning a different detector for every image position, the network learns one local detector and applies it everywhere. This drastically reduces parameter count, allows the same pattern to be detected at different locations, and gives convolution its translation-equivariant structure. For a standard convolution, the weight count is `K_h × K_w × C_in × C_out`, independent of the number of spatial output positions, although larger feature maps still require more computation.**

---

# ⭐ Remember These 7 Things

1. 🔁 Same filter weights are reused across H/W
2. 🎛️ Different filters still have different weights
3. 🌈 The entire multi-channel filter is shared spatially
4. 🧮 H/W do not directly appear in convolution parameter count
5. ⚡ Larger H/W increases computation, not independent filter parameters
6. 📍 Sharing supports translation-equivariant behavior
7. 🗺️ Feature maps still preserve where responses occur

---

# ⭐ Golden Rule

> **Learn the pattern once, reuse the detector everywhere: CNN parameter sharing means one filter has one set of weights regardless of how many spatial locations it scans.**

---

# ➡️ Next Topic

## 📘 Receptive Field

Next we will study:

* 👁️ What region of the input influences one CNN activation
* 📐 Local receptive fields
* 🏗️ How receptive field grows with depth
* 🎛️ Effect of kernel size
* ➡️ Effect of stride
* 🏊 Effect of pooling
* 🧠 Why deeper neurons can represent larger spatial context
