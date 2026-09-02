# ⚡ Cheat Sheet — CNN vs Fully Connected Neural Networks

> **Core Idea:** Fully connected networks learn global independent connections, while CNNs exploit **local connectivity, spatial parameter sharing, and spatial structure**.

---

# 🎯 1. Main Difference

### Fully Connected

```text
Every Input
↓
Connected to Every Output Neuron
```

### CNN

```text
Local Patch
↓
Shared Filter
↓
Feature Map
```

Shortcut:

```text
Dense → Global + Independent Weights
CNN   → Local + Shared Weights
```

---

# 🖼️ 2. Input Representation

### Dense Network

Image is often:

```text
H×W×C
↓
Flatten
↓
Vector
```

Example:

`28×28 → 784`

### CNN

Keeps:

`H×W×C`

spatial structure through feature maps.

---

# ⚠️ 3. Flattening Precision

Do NOT say:

> Flattening destroys all spatial information.

Better:

> Flattening preserves values and ordering, but removes explicit spatial locality from the architecture.

---

# 🔍 4. Local Connectivity

CNN output activation sees:

> only a local region initially.

Example:

`3×3 kernel → 3×3 receptive field`

Dense neuron:

> can directly connect to the entire flattened input.

---

# 🔄 5. Parameter Sharing

CNN:

```text
One Filter
↓
Left
Center
Right
Everywhere
```

Same learned weights are reused spatially.

This enables:

> **learn once, detect everywhere**

---

# ⚠️ 6. Locality ≠ Sharing

### Local Connectivity

Output uses a local patch.

### Parameter Sharing

Same weights are reused at different locations.

They are:

> **different concepts**

---

# 🧠 7. CNN Inductive Biases

Important CNN assumptions:

1. Nearby values matter.
2. Useful patterns can repeat spatially.
3. The same detector can often work at different locations.

These assumptions are especially useful for:

> images and spatial data.

---

# 🧮 8. Dense Parameter Formula

For:

`N_in → N_out`

Weights:

`N_in × N_out`

With bias:

`N_in × N_out + N_out`

Dense parameter count depends strongly on:

> total flattened input size.

---

# 🧮 9. Conv Parameter Formula

Standard Conv2D:

`K_h × K_w × C_in × C_out`

With bias:

`+ C_out`

So:

> H and W do not directly multiply Conv parameter count.

---

# ⭐ 10. Why Conv Parameters Ignore H/W

Because:

> the same filter is reused spatially.

Increasing image resolution means:

> more filter applications

not:

> more filter parameters.

---

# 🧮 11. Parameter Example

Input:

`32×32×3`

### Dense → 64 neurons

`3072×64 + 64`

`= 196,672`

### Conv `3×3`, 64 filters

`3×3×3×64 + 64`

`= 1,792`

Huge difference in:

> parameter count.

But the outputs are NOT equivalent.

---

# ⚠️ 12. Dense vs Conv Output Warning

Dense:

`3072 → 64`

produces:

`64`

values.

Conv:

`32×32×3 → 32×32×64`

produces:

`65,536`

activations.

So:

> parameter comparison ≠ equivalent computation.

---

# 🗺️ 13. Spatial Structure

CNN output remains:

`H_out × W_out × C_out`

So feature location is approximately retained.

Dense representation typically becomes:

> a vector.

---

# 🎯 14. Translation Equivariance

Standard convolution tends to satisfy:

```text
Shift Input
↓
Feature Response Shifts
```

This is:

> **translation equivariance**

---

# ⚠️ 15. Equivariance ≠ Invariance

### Equivariance

Input shifts → feature response shifts.

### Invariance

Input shifts → output stays unchanged.

Convolution itself is mainly:

> equivariant

not perfectly invariant.

---

# 👁️ 16. Receptive Field

### Dense

A neuron can have:

> global connectivity immediately.

### CNN

Starts:

> local.

Then:

```text
Local RF
↓
Larger RF
↓
Broader Context
```

through depth.

---

# 🧠 17. CNN Feature Hierarchy

Typical progression:

```text
Pixels
↓
Local Patterns
↓
Pattern Combinations
↓
Shapes / Textures
↓
Higher-Level Features
```

Enabled by:

* locality
* depth
* receptive-field growth
* nonlinearities

---

# 🌍 18. Why CNNs Often Generalize Better on Images

CNNs use:

```text
Locality
+
Weight Sharing
+
Spatial Structure
```

This reduces:

> unnecessary independent parameters.

When assumptions match the task:

* sample efficiency improves
* parameter efficiency improves
* generalization can improve

---

# ⚖️ 19. Inductive Bias Trade-Off

CNN:

```text
More Structural Assumptions
↓
Less Positional Freedom
↓
Better Efficiency When Assumptions Match
```

Dense:

```text
Fewer Spatial Assumptions
↓
More Independent Weights
↓
Greater Raw Flexibility
```

So:

> CNN is not universally better.

---

# ⚠️ 20. When Weight Sharing Can Hurt

Weight sharing assumes:

> same local transformation is useful everywhere.

This may be restrictive when:

> absolute position strongly changes the meaning of a pattern.

So sharing is:

> a useful bias, not a universal rule.

---

# 📊 21. When Dense Networks Are Natural

Dense networks are often suitable for:

* tabular data
* compact feature vectors
* global feature combinations
* non-spatial inputs

Example:

```text
Age
Salary
Credit Score
Balance
Tenure
```

No meaningful reason exists to slide a spatial kernel across them.

---

# 🖼️ 22. When CNNs Are Natural

CNNs are natural for:

* images
* spatial grids
* spectrograms
* signals
* some time series
* volumetric data

when:

> local repeated patterns matter.

---

# 🔢 23. Conv Dimensions

### Conv1D

Often:

* signals
* sequences
* time series

### Conv2D

Often:

* images

### Conv3D

Often:

* videos
* medical volumes

Core principles remain:

> locality + sharing.

---

# 🎯 24. CNN + Dense Together

CNN does NOT mean:

> no dense layers.

Common classification architecture:

```text
Image
↓
CNN Backbone
↓
GAP
↓
Linear Layer
↓
Logits
```

Think:

```text
CNN   → Feature Extraction
Dense → Final Task Mapping
```

---

# 🌍 25. GAP vs Flatten

### Flatten

`H×W×C → HWC`

Usually ties the dense layer to:

> fixed spatial dimensions.

### GAP

`H×W×C → C`

Can provide:

> greater spatial-size flexibility.

---

# 🧮 26. CNN Compute

Approximate Conv multiplications:

`H_out × W_out × K_h × K_w × C_in × C_out`

Therefore:

> CNN can have few parameters but still high compute.

---

# ⭐ 27. Parameters vs Compute

Remember:

```text
Weight Sharing
↓
Parameter Count ↓
```

But:

```text
Filter Applied Many Times
↓
Compute Can Still Be High
```

So:

> **few parameters ≠ few operations**

---

# 📦 28. Memory Trade-Off

Dense networks may have:

> huge parameter memory.

CNNs often reduce parameter memory but may have:

> huge activation tensors.

Example:

`224×224×64`

contains:

`3,211,264 activations`

---

# 📐 29. Effect of Resolution

Same Conv:

`3×3, 64→128`

on:

`56×56`

and:

`112×112`

has:

> same trainable parameter count.

But:

`112×112`

requires much more computation.

---

# 🧠 30. Universal Approximation Connection

Dense networks may theoretically represent image functions.

But:

```text
Can Represent
≠
Can Represent Efficiently
≠
Can Train Easily
≠
Can Generalize Well
```

CNNs give useful:

> problem-specific structure.

---

# 🔗 31. Backpropagation Difference

Dense:

```text
Different Connections
↓
Different Weights
↓
Separate Parameter Gradients
```

CNN:

```text
Many Spatial Locations
↓
Same Shared Weight
↓
Gradient Contributions Sum
```

So sharing matters in:

> forward and backward passes.

---

# ⚡ 32. Dense vs CNN Table

| Concept                    | Fully Connected          | CNN                     |
| -------------------------- | ------------------------ | ----------------------- |
| Connectivity               | Global                   | Local initially         |
| Weight sharing             | Usually no               | Spatial                 |
| Input form                 | Usually vector           | Spatial tensor          |
| H/W explicitly preserved   | Usually no               | Yes                     |
| Params grow with H/W       | Often                    | Not directly            |
| Initial RF                 | Global                   | Local                   |
| Pattern reuse              | Not built in             | Built in                |
| Translation structure      | Weak                     | Strong equivariant bias |
| Image parameter efficiency | Usually poor             | Usually strong          |
| Best fit                   | Compact/non-spatial data | Local/spatial data      |

---

# 🚫 Interview Traps

### ❌ Dense networks cannot process images

They can.

---

### ❌ Flatten deletes spatial information completely

Too strong.

---

### ❌ CNN has no dense layers

False.

---

### ❌ Local connectivity = parameter sharing

False.

---

### ❌ Conv parameter count grows with H/W

Not directly.

---

### ❌ Conv compute does not depend on H/W

False.

---

### ❌ CNN automatically means low compute

False.

---

### ❌ Convolution gives perfect translation invariance

False.

Primarily equivariance.

---

### ❌ CNN is always better than Dense

False.

Architecture must match data structure.

---

### ❌ Same parameter count means same runtime

False.

---

# 🎤 30-Second Interview Answer

> **A fully connected network typically flattens the image and learns independent global connections between inputs and neurons, which can lead to very large parameter counts and weak spatial inductive bias. A CNN instead uses local receptive fields and shares learned filters across spatial positions. This preserves spatial feature maps, allows reusable pattern detection, and greatly improves parameter efficiency. CNNs are therefore well suited to image-like data, although they can still be computationally expensive and are not universally better for every type of input.**

---

# 🧠 Final Mental Model

```text
FULLY CONNECTED

Image
↓
Flatten
↓
Global Connections
↓
Independent Weights
↓
Large Parameter Count


CNN

Image
↓
Local Patches
↓
Shared Filters
↓
Spatial Feature Maps
↓
Growing Receptive Field
↓
Feature Hierarchy
↓
Global Representation
```

---

# ⭐ Remember These 8 Things

1. 🔍 CNN = local connectivity.
2. 🔄 CNN = spatial parameter sharing.
3. 🖼️ Dense usually flattens; CNN preserves spatial tensors.
4. 🧮 Dense params can scale with H×W; Conv params do not directly.
5. ⚡ Conv compute still scales with spatial resolution.
6. 🎯 Convolution gives translation-equivariant structure, not perfect invariance.
7. 🧠 Dense networks can learn images, but CNN inductive bias often makes it much more efficient.
8. ⚖️ Choose architecture based on data structure and task.

---

# ⭐ Golden Rule

> **Dense networks learn many independent global connections; CNNs learn local filters once and reuse them across space.**
