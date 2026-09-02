# 📘 Lecture — Output Dimension in CNNs

> **Core Idea:** The output spatial size of a convolution depends on four things: input size, kernel size, padding, and stride.

---

# 🎯 Learning Objectives

By the end of this lecture, you should understand:

* 📐 The output-dimension formula
* 🧠 Where the formula comes from
* 🧮 How to calculate output height and width
* 0️⃣ How no padding affects output size
* 🧱 How padding affects output size
* ➡️ How stride affects output size
* 🟰 How same-size outputs are created
* 📉 How downsampling happens
* ↔️ How to handle non-square inputs
* 📦 Why output channels are calculated separately
* ⚠️ Why `floor()` is used
* 🎤 How to solve CNN shape questions quickly in interviews

---

# 🧭 1. The Question We Are Solving

We already understand:

### 🎛️ Kernel Size

How much of the input the filter looks at.

### ➡️ Stride

How far the filter moves.

### 🧱 Padding

How much extra boundary is added.

Now we can answer:

> **How many valid positions can the filter occupy?**

That number determines the output size.

---

# 📐 2. The Main Formula

For one spatial dimension:

`Output = floor((Input + 2P - K) / S) + 1`

Where:

* `Input` = input size
* `K` = kernel size
* `P` = padding
* `S` = stride

For height:

`H_out = floor((H_in + 2P_h - K_h) / S_h) + 1`

For width:

`W_out = floor((W_in + 2P_w - K_w) / S_w) + 1`

---

# 🧠 3. Where Does the Formula Come From?

Start with:

`Input`

If we add padding on both sides:

`Input + 2P`

Now place a kernel of size:

`K`

The first valid position consumes:

`K`

positions.

So the remaining movement space is:

`Input + 2P - K`

Then divide by stride:

`(Input + 2P - K) / S`

This tells us how many jumps we can make.

But we must also count the:

> **first filter position**

So we add:

`+1`

Therefore:

`Output = floor((Input + 2P - K) / S) + 1`

---

# 🧩 4. Simplest Example

Input:

`5`

Kernel:

`3`

Padding:

`0`

Stride:

`1`

Formula:

`Output = floor((5 + 0 - 3) / 1) + 1`

`= floor(2) + 1`

`= 3`

So in 2D:

Input:

`5 × 5`

Kernel:

`3 × 3`

Output:

> **3 × 3**

---

# 🔍 5. Match It with Filter Positions

For one dimension:

Input:

`1 2 3 4 5`

Kernel size:

`3`

Stride:

`1`

Positions:

`[1 2 3]`

`[2 3 4]`

`[3 4 5]`

There are:

`3`

valid positions.

The formula gives exactly:

`3`

---

# 🧱 6. Example with Padding

Input:

`5`

Kernel:

`3`

Padding:

`1`

Stride:

`1`

Formula:

`Output = floor((5 + 2×1 - 3) / 1) + 1`

`= floor(4) + 1`

`= 5`

So in 2D:

`5 × 5`

↓ `3×3`, P=1, S=1

`5 × 5`

The spatial size is preserved.

---

# 🟰 7. Same-Size Case

For odd kernel sizes with:

`stride = 1`

we often choose:

`P = (K - 1) / 2`

Example:

Kernel:

`3`

Then:

`P = (3 - 1)/2 = 1`

Example:

Input:

`32`

Kernel:

`3`

Padding:

`1`

Stride:

`1`

Output:

`floor((32 + 2 - 3)/1) + 1`

`= 31 + 1`

`= 32`

So:

> **Input size = Output size**

---

# 📏 8. Another Same-Size Example

Input:

`100`

Kernel:

`5`

Stride:

`1`

To preserve size:

`P = (5 - 1)/2`

`= 2`

Then:

`Output = floor((100 + 4 - 5)/1) + 1`

`= 99 + 1`

`= 100`

So:

`5 × 5 kernel`

with:

`padding = 2`

preserves spatial size for stride 1.

---

# ➡️ 9. Example with Stride 2

Input:

`7`

Kernel:

`3`

Padding:

`0`

Stride:

`2`

Formula:

`Output = floor((7 - 3)/2) + 1`

`= floor(4/2) + 1`

`= 2 + 1`

`= 3`

So in 2D:

Input:

`7 × 7`

Kernel:

`3 × 3`

Stride:

`2`

Output:

> **3 × 3**

---

# 🔍 10. Why Does Stride 2 Give 3 Positions?

Input:

`1 2 3 4 5 6 7`

Kernel size:

`3`

Stride:

`2`

Windows:

`[1 2 3]`

`[3 4 5]`

`[5 6 7]`

So:

`3`

valid positions.

---

# ⚠️ 11. Why Do We Need floor()?

Consider:

Input:

`8`

Kernel:

`3`

Padding:

`0`

Stride:

`2`

Formula before floor:

`(8 - 3)/2 + 1`

`= 5/2 + 1`

`= 2.5 + 1`

Obviously we cannot have:

`3.5 output positions`

The filter either fits or it does not.

So:

`floor(2.5) + 1`

`= 2 + 1`

`= 3`

Therefore:

> **floor() removes an incomplete final filter position.**

---

# 🧠 12. The Incomplete Window Intuition

Input:

`1 2 3 4 5 6 7 8`

Kernel:

`3`

Stride:

`2`

Valid windows:

`[1 2 3]`

`[3 4 5]`

`[5 6 7]`

The next would start too late and cannot fit a complete kernel.

So it is discarded.

That is exactly why:

> `floor()`

appears in the formula.

---

# 🧮 13. Full 2D Example

Suppose:

Input:

`32 × 32`

Kernel:

`5 × 5`

Padding:

`0`

Stride:

`1`

Height:

`H_out = floor((32 - 5)/1) + 1`

`= 27 + 1`

`= 28`

Width:

`W_out = 28`

So output:

> **28 × 28**

---

# 🧱 14. Add Padding to the Same Example

Input:

`32 × 32`

Kernel:

`5 × 5`

Padding:

`2`

Stride:

`1`

Height:

`H_out = floor((32 + 4 - 5)/1) + 1`

`= 31 + 1`

`= 32`

Width:

`32`

Output:

> **32 × 32**

---

# 📉 15. Example with Padding and Stride Together

Input:

`32`

Kernel:

`3`

Padding:

`1`

Stride:

`2`

Formula:

`Output = floor((32 + 2 - 3)/2) + 1`

`= floor(31/2) + 1`

`= 15 + 1`

`= 16`

So:

`32 × 32`

becomes:

> **16 × 16**

This is a common downsampling pattern.

---

# 🎯 16. A Very Common CNN Pattern

You will often see:

Kernel:

`3 × 3`

Padding:

`1`

Stride:

`2`

Then:

Spatial size roughly halves.

Example:

`64 × 64`

↓

`32 × 32`

↓

`16 × 16`

↓

`8 × 8`

This is one common way CNNs progressively reduce spatial resolution.

---

# ↔️ 17. Height and Width Are Calculated Separately

Suppose:

Input:

`28 × 40`

Kernel:

`3 × 5`

Padding:

`1 × 2`

Stride:

`1 × 2`

Then calculate height separately:

`H_out = floor((28 + 2×1 - 3)/1) + 1`

`= floor(27) + 1`

`= 28`

Width:

`W_out = floor((40 + 2×2 - 5)/2) + 1`

`= floor(39/2) + 1`

`= 19 + 1`

`= 20`

So output spatial size:

> **28 × 20**

---

# 🧠 18. Non-Square Inputs Are Not Special

The same formula works independently for both dimensions.

So for:

`H_in ≠ W_in`

or:

`K_h ≠ K_w`

just calculate:

Height
and
Width

separately.

No new concept is needed.

---

# 📦 19. What About Output Channels?

Output height and width depend on:

* Input size
* Kernel size
* Padding
* Stride

But output channels depend on:

> **number of filters**

Example:

Input:

`32 × 32 × 3`

Kernel:

`3 × 3`

Padding:

`1`

Stride:

`1`

Filters:

`64`

Spatial output:

`32 × 32`

Channels:

`64`

Final output:

> **32 × 32 × 64**

---

# ⭐ 20. Shape Formula Including Channels

Conceptually:

Input:

`H_in × W_in × C_in`

Conv layer:

* Kernel = `K_h × K_w`
* Padding = `P_h, P_w`
* Stride = `S_h, S_w`
* Filters = `C_out`

Output:

`H_out × W_out × C_out`

where:

`H_out = floor((H_in + 2P_h - K_h)/S_h) + 1`

`W_out = floor((W_in + 2P_w - K_w)/S_w) + 1`

and:

> `C_out = number of filters`

---

# 🧮 21. Interview Example 1

Input:

`224 × 224 × 3`

Conv:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `1`
* Filters = `64`

Height:

`floor((224 + 2 - 3)/1) + 1`

`= 223 + 1`

`= 224`

Width:

`224`

Channels:

`64`

Output:

> **224 × 224 × 64**

---

# 🧮 22. Interview Example 2

Input:

`224 × 224 × 64`

Conv:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `2`
* Filters = `128`

Height:

`floor((224 + 2 - 3)/2) + 1`

`= floor(223/2) + 1`

`= 111 + 1`

`= 112`

Output:

> **112 × 112 × 128**

---

# 🧮 23. Interview Example 3

Input:

`28 × 28 × 32`

Conv:

* Kernel = `5 × 5`
* Padding = `0`
* Stride = `1`
* Filters = `64`

Height:

`floor((28 - 5)/1) + 1`

`= 23 + 1`

`= 24`

So output:

> **24 × 24 × 64**

---

# 🧮 24. Interview Example 4

Input:

`64 × 64 × 16`

Conv:

* Kernel = `7 × 7`
* Padding = `3`
* Stride = `2`
* Filters = `32`

Height:

`floor((64 + 6 - 7)/2) + 1`

`= floor(63/2) + 1`

`= 31 + 1`

`= 32`

Output:

> **32 × 32 × 32**

---

# 🧠 25. Quick Interview Calculation Method

When given a CNN shape problem:

### Step 1

Write:

`Input = H × W × C_in`

### Step 2

Identify:

* `K`
* `P`
* `S`
* Number of filters

### Step 3

Calculate:

`H_out`

### Step 4

Calculate:

`W_out`

### Step 5

Set:

`C_out = number of filters`

Then combine:

`H_out × W_out × C_out`

---

# ⚠️ 26. Common Mistake — Forgetting +1

Wrong:

`Output = (Input + 2P - K)/S`

Correct:

`Output = floor((Input + 2P - K)/S) + 1`

Why?

Because:

> **the first kernel position must also be counted.**

---

# ⚠️ 27. Common Mistake — Forgetting 2P

Padding is added on:

* left and right
* top and bottom

So for one dimension:

total padding contribution is:

`2P`

not:

`P`

Therefore:

`Input + 2P`

---

# ⚠️ 28. Common Mistake — Using Input Channels in Spatial Formula

Input channels do NOT appear in:

`H_out`

or:

`W_out`

You do not calculate:

`(Input + channels + padding - kernel)/stride`

Channels are handled separately.

---

# ⚠️ 29. Common Mistake — Thinking Output Channels Come from Input Channels

Example:

Input:

`32 × 32 × 3`

Filters:

`128`

Output channels:

> `128`

not:

`3`

So:

`C_in`

and:

`C_out`

serve different roles.

---

# 🏗️ 30. Layer-by-Layer Shape Tracking

Suppose:

Input:

`224 × 224 × 3`

---

### Conv 1

K = 3
P = 1
S = 1
Filters = 64

Output:

`224 × 224 × 64`

---

### Conv 2

K = 3
P = 1
S = 2
Filters = 128

Output:

`112 × 112 × 128`

---

### Conv 3

K = 3
P = 1
S = 2
Filters = 256

Output:

`56 × 56 × 256`

Notice the pattern:

Spatial Resolution
↓ decreases

Channels
↑ increase

This is common in CNN design.

---

# 📊 31. Quick Comparison

| Setting Change      | Typical Effect                      |
| ------------------- | ----------------------------------- |
| Larger Kernel       | Smaller output if padding unchanged |
| More Padding        | Larger/preserved output             |
| Larger Stride       | Smaller output                      |
| More Filters        | More output channels                |
| More Input Channels | Does not directly change H/W        |

---

# 🧠 32. Complete Mental Story

Start with:

Input Size

↓

Padding adds usable boundary:

`Input + 2P`

↓

Kernel occupies:

`K`

↓

Remaining movement range:

`Input + 2P - K`

↓

Stride determines number of jumps:

`/ S`

↓

Floor removes incomplete final position

↓

`+1` counts the first position

↓

Output Dimension

---

# 🎤 30-Second Interview Answer

> **The spatial output size of a convolution is calculated as `floor((Input + 2P - K) / S) + 1`. Padding increases the usable input boundary, kernel size determines how much space each convolution window occupies, and stride determines how far the filter moves. Height and width are calculated separately, while output channels are simply determined by the number of filters.**

---

# 📌 Key Takeaways

* 📐 Main formula:

`Output = floor((Input + 2P - K) / S) + 1`

* 🧱 Padding contributes `2P`
* 🎛️ Larger kernel usually reduces output size
* ➡️ Larger stride reduces output size
* 🟰 Appropriate padding can preserve spatial size
* ⚠️ `floor()` removes incomplete final windows
* ➕ `+1` counts the first valid filter position
* ↔️ Calculate height and width separately
* 📦 Output channels = number of filters
* 🧠 Input channels do not directly affect output H/W
* 🎯 Always calculate spatial shape and channel shape separately

---

# ⭐ Golden Rule

> **Kernel size, stride, and padding determine the output height and width; the number of filters determines the output channels.**

---

# ➡️ Next Topic

## 📘 Multiple Filters and Channels

Next we will study:

* 🌈 How convolution works across multiple input channels
* 🎛️ Why one filter spans all input channels
* ➕ How channel-wise contributions are summed
* 📦 Why one full filter still produces one feature map
* 🧮 Parameter-count calculations for multi-channel convolution
