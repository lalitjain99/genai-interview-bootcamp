# Learning Rate Cheat Sheet

Problem

Gradient tells direction.

Need step size.

Solution

Learning Rate (η)

Formula

W = W − η∇L

Small η

✔ Stable

❌ Slow

Large η

✔ Fast initially

❌ Overshooting

Remember

Gradient → Direction

Learning Rate → Step Size

Interview Trap

Gradient already has magnitude.

Learning Rate decides how much we trust that magnitude.

Limitation

One global learning rate cannot suit every parameter.

Leads to AdaGrad.