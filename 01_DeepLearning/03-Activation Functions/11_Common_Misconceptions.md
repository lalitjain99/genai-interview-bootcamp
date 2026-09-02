# Common Mistakes

❌ ReLU is linear.

Correct:

Each region is linear.

The complete function is not.

---------------------------------

❌ GELU is just smoother ReLU.

Correct:

GELU introduces probabilistic gating.

---------------------------------

❌ Sigmoid outputs probabilities.

Correct:

Sigmoid outputs values between 0 and 1.

Only in specific contexts are these interpreted as probabilities.

---------------------------------

❌ Softmax is an activation like ReLU.

Correct:

Softmax converts logits into a probability distribution over classes.

---------------------------------

❌ Leaky ReLU completely fixes Dying ReLU.

Correct:

It mitigates the problem but doesn't eliminate all training issues.