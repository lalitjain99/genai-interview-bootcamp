# 📘 Lecture — Image Representation in CNNs

> **Core Idea:** Before a CNN can detect edges, textures, or objects, an image must first be represented as numbers. CNNs process these numbers while preserving the image's spatial structure.

---

# 🎯 Learning Objectives

By the end of this lecture, you should understand:

- 🖼️ What a pixel represents
- ⚫ How grayscale images are represented
- 🌈 How RGB images are represented
- 📐 What `Height × Width × Channels` means
- 🧱 Why an image is called a tensor
- 📦 How batches of images are represented
- 🔄 Common tensor layouts
- ⚠️ Difference between image channels and CNN filters
- 🧠 How one CNN layer's output becomes the next layer's input

---

# 🖼️ 1. What Is an Image to a Neural Network?

Humans see:

> cat, dog, car, tree

A neural network sees:

> **numbers**

An image is stored as a grid of pixel values.

For example, a small grayscale image might look like:

10   20   30  
40   50   60  
70   80   90

Each number represents one pixel intensity.

So:

Image  
↓  
Pixel Grid  
↓  
Numbers  
↓  
CNN

---

# ⚫ 2. Grayscale Image Representation

A grayscale image contains only one intensity value per pixel.

For an 8-bit grayscale image, values are commonly:

`0 → black`

`255 → white`

Values between them represent different shades of gray.

Example:

0     50    100  
150   200   255  
100   50     0

If the image has:

Height = 3  
Width = 3

then its shape can be written as:

`3 × 3`

Conceptually:

`Height × Width`

---

# 🧮 3. Number of Values in a Grayscale Image

Suppose:

Height = 100  
Width = 100

Total pixel values:

`100 × 100 = 10,000`

So a `100 × 100` grayscale image contains:

> **10,000 intensity values**

---

# 🌈 4. RGB Image Representation

Color images commonly use three channels:

- 🔴 Red
- 🟢 Green
- 🔵 Blue

Each pixel therefore contains three numbers.

For example, one pixel might be:

`R = 255`  
`G = 0`  
`B = 0`

This represents strong red.

Another pixel:

`R = 0`  
`G = 255`  
`B = 0`

represents green.

So an RGB image has shape:

`Height × Width × 3`

---

# 🧱 5. Think of RGB as Three Stacked Images

Suppose we have:

`4 × 4 RGB image`

Instead of one matrix, we effectively have three matrices:

🔴 Red Channel  
`4 × 4`

🟢 Green Channel  
`4 × 4`

🔵 Blue Channel  
`4 × 4`

Stack them together:

Red  
+  
Green  
+  
Blue  
↓  
RGB Image Tensor

Shape:

`4 × 4 × 3`

---

# 📐 6. Understanding Height × Width × Channels

Suppose:

`224 × 224 × 3`

This means:

- Height = `224`
- Width = `224`
- Channels = `3`

The total number of values is:

`224 × 224 × 3`

= `150,528`

So the CNN receives:

> **150,528 numerical values**

while still preserving their spatial/channel organization.

---

# 🧠 7. What Exactly Is a Channel?

A channel is one component of the image representation.

For RGB:

Channel 1 → Red  
Channel 2 → Green  
Channel 3 → Blue

Each channel has its own:

`Height × Width`

matrix.

So:

RGB Image  
↓  
3 Channels  
↓  
Each channel contains a 2D pixel grid

---

# 🧱 8. Why Do We Call an Image a Tensor?

A tensor is simply a multi-dimensional array.

Examples:

Scalar:

`5`

→ 0-dimensional

Vector:

`[1, 2, 3]`

→ 1-dimensional

Matrix:

3 × 3 grid

→ 2-dimensional

RGB Image:

`Height × Width × Channels`

→ 3-dimensional tensor

So:

> **An RGB image is naturally represented as a 3D tensor.**

---

# ⚫ 9. Grayscale Can Also Include a Channel Dimension

A grayscale image may be described as:

`Height × Width`

But in CNN frameworks, it is often represented explicitly with one channel:

`Height × Width × 1`

For example:

`28 × 28 × 1`

means:

- Height = 28
- Width = 28
- Channels = 1

---

# 📦 10. What Happens When We Process Multiple Images?

During training, we rarely process only one image at a time.

We usually process a:

> **Batch**

Suppose:

Batch size = 32

Each image:

`224 × 224 × 3`

Then the overall data contains another dimension:

`Batch × Height × Width × Channels`

Example:

`32 × 224 × 224 × 3`

So:

32 Images  
↓  
Each 224 × 224  
↓  
Each with 3 channels

---

# 🔄 11. Tensor Layout Can Differ Between Frameworks

Not every framework stores dimensions in the same order.

A common image layout is:

`H × W × C`

meaning:

Height × Width × Channels

Another common layout is:

`C × H × W`

meaning:

Channels × Height × Width

For a batch, you may see:

`N × H × W × C`

or:

`N × C × H × W`

where:

`N = Batch Size`

---

# 🐍 12. Common PyTorch Format

PyTorch commonly uses:

`N × C × H × W`

Example:

`32 × 3 × 224 × 224`

means:

- Batch = 32
- Channels = 3
- Height = 224
- Width = 224

One image therefore has shape:

`3 × 224 × 224`

---

# ⚠️ 13. Channels Are NOT Filters

This is an important distinction.

For an RGB image:

`3 channels`

means:

Red  
Green  
Blue

These are part of the **input data**.

A CNN may then have:

`32 filters`

Those filters are:

> **Learned pattern detectors**

So:

Input Channels  
≠  
Number of Filters

Example:

Input:

`224 × 224 × 3`

CNN Layer:

`32 filters`

Output:

> **32 feature maps**

The output will therefore have:

`32 channels`

---

# 🧠 14. Feature Maps Become Channels for the Next Layer

This is one of the most important ideas.

Suppose the original image has:

`3 channels`

Then the first convolutional layer uses:

`32 filters`

It produces:

`32 feature maps`

These 32 feature maps are stacked together.

So the output becomes something like:

`Height × Width × 32`

That output becomes the input to the next convolutional layer.

Therefore:

RGB Image  
`3 channels`  
↓  
Conv Layer with 32 filters  
↓  
`32 feature maps`  
↓  
Next layer receives  
`32 channels`

So after the first layer, channels are no longer necessarily:

Red / Green / Blue.

They become:

> **Learned feature channels**

---

# 🎛️ 15. Example

Suppose input shape is:

`32 × 32 × 3`

The convolutional layer contains:

`16 filters`

Each filter scans the image.

Then:

Filter 1 → Feature Map 1  
Filter 2 → Feature Map 2  
...  
Filter 16 → Feature Map 16

So the output has:

`16 channels`

Conceptually:

Input  
`32 × 32 × 3`  
↓  
16 Filters  
↓  
16 Feature Maps  
↓  
Output  
`H_out × W_out × 16`

We will learn how to calculate:

`H_out`

and:

`W_out`

when we cover stride and padding.

---

# 🔍 16. Why Preserve the Tensor Structure?

A CNN does not immediately flatten the image because we want to preserve:

- 📍 Pixel location
- 🧩 Neighborhood relationships
- 🌈 Channel information
- 🗺️ Spatial patterns

This allows convolutional filters to operate on local regions.

So instead of:

Image  
↓  
Flatten Immediately

CNN keeps:

Height  
×  
Width  
×  
Channels

for most of the feature-extraction process.

---

# 🧠 17. The Most Important Transition

At the beginning:

Channels represent:

🔴 Red  
🟢 Green  
🔵 Blue

After convolution:

Channels represent:

🎛️ Learned Feature Map 1  
🎛️ Learned Feature Map 2  
🎛️ Learned Feature Map 3  
...  

So:

Input Channels  
↓  
Raw visual information

Deeper Channels  
↓  
Learned visual features

---

# ⚡ Quick Summary

| Concept | Meaning |
|---|---|
| Pixel | Numerical intensity/color value |
| Grayscale | One channel |
| RGB | Three channels |
| Image Shape | `H × W × C` |
| Tensor | Multi-dimensional numerical array |
| Batch | Multiple images processed together |
| PyTorch Layout | Commonly `N × C × H × W` |
| Input Channel | Existing image/feature dimension |
| Filter | Learned pattern detector |
| Feature Map | Output of one filter |
| Output Channels | Number of feature maps |

---

# 🧠 Mental Model

Raw RGB Image  
↓  
`H × W × 3`  
↓  
CNN Filters  
↓  
Multiple Feature Maps  
↓  
`H_out × W_out × Number_of_Filters`  
↓  
These Feature Maps Become  
Channels for the Next Layer

---

# 🎤 30-Second Interview Answer

> **An image is represented as a tensor containing pixel values. A grayscale image has one channel, while an RGB image has three channels representing red, green, and blue. Its shape can be represented as height × width × channels. During CNN processing, convolutional filters create feature maps, and those feature maps become the channels of the next layer. Frameworks may use different tensor layouts, such as H×W×C or C×H×W.**

---

# 📌 Key Takeaways

- 🖼️ Images are numerical tensors
- ⚫ Grayscale usually has 1 channel
- 🌈 RGB has 3 channels
- 📐 Image shape contains height, width, and channels
- 📦 Batches add another dimension
- 🔄 Frameworks can use different dimension orders
- 🎛️ Filters and channels are different concepts
- 🗺️ One filter produces one feature map
- 🧱 Multiple feature maps become multiple output channels
- 🔗 Those output channels become input channels for the next convolutional layer

---

# ⭐ Golden Rule

> **An image starts as spatial pixel channels, and as it moves through a CNN, those channels gradually become learned feature representations.**

---

# ➡️ Next Topic

## 📘 Convolution Operation

Next, we will answer:

> **How does a filter actually take a small region of an image and convert it into one feature-map value?**

That will lead us to the core convolution calculation:

Local Region  
×  
Kernel  
↓  
Element-wise Multiplication  
↓  
Sum  
↓  
One Output Value