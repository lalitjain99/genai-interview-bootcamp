# 🚫 Common Mistakes — Output Dimension in CNNs

> **Core Idea:** Most output-shape mistakes come from forgetting `2P`, `+1`, `floor()`, mixing channels into the spatial formula, or skipping layer-by-layer calculations.

---

## ❌ Mistake 1: Forgetting the `+1`

### Wrong

`Output = floor((Input + 2P - K) / S)`

### Correct

`Output = floor((Input + 2P - K) / S) + 1`

Why?

Because the division counts how many stride jumps are possible.

We still need to count:

> **the first valid filter position**

---

## ❌ Mistake 2: Using `P` instead of `2P`

### Wrong

`Input + P`

### Correct

`Input + 2P`

Padding is added on both sides of a dimension.

For width:

Left Padding
+
Right Padding

For height:

Top Padding
+
Bottom Padding

So total contribution is:

> **2P**

---

## ❌ Mistake 3: Forgetting `floor()`

### Wrong

If the calculation gives:

`3.5`

then output size is:

`3.5`

### Correct

Output positions must be whole numbers.

So:

`floor(3.5) = 3`

The incomplete final convolution window is ignored.

---

## ❌ Mistake 4: Rounding Instead of Flooring

### Wrong

`2.7 → 3`

because normal rounding gives 3.

### Correct

Convolution uses:

> **floor**

So:

`floor(2.7) = 2`

We do not round to the nearest integer.

We count only fully valid positions.

---

## ❌ Mistake 5: Including Input Channels in the Spatial Formula

### Wrong

For:

`64 × 64 × 32`

write:

`floor((64 + 32 + 2P - K)/S) + 1`

### Correct

Input channels do not belong in:

`H_out`

or:

`W_out`

The spatial formula only uses:

* Input height/width
* Kernel size
* Padding
* Stride

Channels are handled separately.

---

## ❌ Mistake 6: Assuming Output Channels Equal Input Channels

### Wrong

Input:

`32 × 32 × 3`

therefore output must also have:

`3 channels`

### Correct

Output channels are determined by:

> **number of filters**

Example:

Input Channels = `3`

Filters = `64`

Output Channels = `64`

---

## ❌ Mistake 7: Thinking More Filters Increase H and W

### Wrong

> Increasing filters from 32 to 128 makes the output spatially larger.

### Correct

Filter count controls:

`C_out`

not:

`H_out` or `W_out`

So:

More Filters
→ More Output Channels

not:

More Filters
→ Larger Spatial Dimensions

---

## ❌ Mistake 8: Assuming Stride 2 Always Gives Exactly Half

### Wrong

> `stride = 2` always means `Output = Input/2`.

### Correct

Stride 2 often roughly halves the spatial size, but the exact result also depends on:

* Kernel size
* Padding
* Input size

Always use:

`floor((Input + 2P - K)/S) + 1`

for explicit numeric padding.

---

## ❌ Mistake 9: Assuming Padding 1 Always Preserves Size

### Wrong

> `padding = 1` always means H/W stay unchanged.

### Correct

That is only true for certain configurations.

A common example is:

* Kernel = `3`
* Stride = `1`
* Padding = `1`

If stride or kernel changes, the result may change.

So:

> **Padding must be interpreted together with kernel size and stride.**

---

## ❌ Mistake 10: Applying the Formula Only Once in a Multi-Layer CNN

### Wrong

Use the original input dimensions for every convolutional layer.

### Correct

Each layer receives:

> **the previous layer's output**

Example:

`224 × 224`

↓ stride 2

`112 × 112`

↓ stride 2

`56 × 56`

So calculations must be done:

> **layer by layer**

---

## ❌ Mistake 11: Forgetting to Calculate Height and Width Separately

### Wrong

> If input height and width differ, use one calculation for both.

### Correct

For non-square inputs, kernels, strides, or padding:

calculate:

`H_out`

and:

`W_out`

separately.

Example:

`28 × 40`

may produce:

`28 × 20`

---

## ❌ Mistake 12: Swapping Height and Width Parameters

### Wrong

Use:

`K_w`

inside the height formula.

### Correct

Use corresponding parameters:

`H_out = floor((H_in + 2P_h - K_h)/S_h) + 1`

`W_out = floor((W_in + 2P_w - K_w)/S_w) + 1`

Keep height-related values together and width-related values together.

---

## ❌ Mistake 13: Thinking Larger Kernel Always Means Smaller Output

### Wrong

> Increasing kernel size always reduces output dimensions.

### Correct

If padding changes at the same time, spatial dimensions may remain the same.

Example:

`3×3, P=1, S=1`

and:

`5×5, P=2, S=1`

can both preserve spatial size.

So:

> **Kernel size cannot be considered independently of padding and stride.**

---

## ❌ Mistake 14: Forgetting That Padding Is Per Side

### Wrong

> `padding = 2` means two total extra positions are added.

### Correct

For symmetric padding:

`2`

is added on each side.

So one dimension gains:

`2 + 2 = 4`

positions.

That is why the formula uses:

`2P`

---

## ❌ Mistake 15: Confusing Output Shape with Parameter Count

### Wrong

> A convolution producing `32 × 32 × 64` has `32×32×64` trainable parameters.

### Correct

`32 × 32 × 64`

describes:

> **activation/output shape**

Parameter count depends on:

`K_h × K_w × C_in × C_out`

plus biases if used.

Output dimension and parameter count are separate calculations.

---

# ⚡ Quick Trap Table

| Wrong Idea                         | Correct Idea                      |
| ---------------------------------- | --------------------------------- |
| Forget `+1`                        | Count the first filter position   |
| Use `P`                            | Use `2P`                          |
| Round result                       | Use `floor()`                     |
| Channels belong in H/W formula     | Channels are separate             |
| `C_out = C_in`                     | `C_out = number of filters`       |
| More filters increase H/W          | They increase channels            |
| Stride 2 always halves exactly     | Exact result depends on K/P/input |
| P=1 always preserves size          | Only for specific settings        |
| Use original input for every layer | Calculate layer by layer          |
| One formula for non-square shapes  | Calculate H and W separately      |
| Larger K always shrinks output     | Padding can compensate            |
| Output shape = parameter count     | Completely different calculations |

---

# 🧠 Final Mental Model

For Spatial Dimensions:

Input
↓
`+ 2P`
↓
`- K`
↓
`/ S`
↓
`floor()`
↓
`+ 1`
↓
`H_out / W_out`

For Channels:

Number of Filters
↓
`C_out`

For Deep CNNs:

Previous Layer Output
↓
Becomes Next Layer Input

---

# ⭐ Golden Rule

> **Never mix spatial geometry with channel count: calculate H and W using `floor((Input + 2P - K)/S) + 1`, take output channels from the filter count, and repeat the process layer by layer.**
