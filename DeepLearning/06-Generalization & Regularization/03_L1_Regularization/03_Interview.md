# 🎯 Interview Questions — L1 Regularization

---

# 📚 Section 1 — Conceptual Questions

---

## Q1. Why do we need Regularization in Machine Learning?

### Answer

Regularization is introduced to improve **generalization**.

Without regularization, a model may memorize the training dataset by learning noise, outliers and accidental correlations.

Although such a model achieves high training accuracy, it performs poorly on unseen data.

Regularization discourages overly complex models by penalizing large weights, forcing the model to learn simpler and more general patterns.

---

## Q2. Why isn't minimizing Prediction Loss alone sufficient?

### Answer

Prediction Loss only measures how well the model fits the training dataset.

The optimizer has no understanding of whether a learned pattern is genuine or simply a coincidence in the training data.

If assigning a very large weight to a noisy feature reduces the training loss, the optimizer will happily do so.

Regularization adds another objective:

> Reduce prediction error while keeping the model as simple as possible.

---

## Q3. What is the intuition behind L1 Regularization?

### Answer

L1 Regularization adds the absolute value of every weight to the loss function.

```text
Total Loss

=

Prediction Loss

+

λ × L1 Loss
```

Large weights now increase the total loss.

The optimizer therefore prefers solutions that achieve good prediction accuracy while keeping weights as small as possible.

---

## Q4. What role does λ (Lambda) play?

### Answer

Lambda controls the trade-off between prediction accuracy and model simplicity.

Small λ

- focuses more on fitting the training data
- higher risk of overfitting

Large λ

- applies stronger weight penalties
- produces simpler models
- excessive values may lead to underfitting

---

# 🧠 Section 2 — Reasoning Questions

---

## Q5. If L1 penalizes every weight, why don't all weights become zero?

### Answer

Because the optimizer minimizes **Total Loss**, not Regularization Loss alone.

Two forces act on every weight.

Prediction Loss tries to preserve weights that improve prediction.

Regularization tries to shrink every weight.

For useful features, reducing the weight significantly increases Prediction Loss.

For unimportant features, reducing the weight barely changes Prediction Loss.

Therefore, useful weights remain relatively large while unimportant weights gradually approach zero.

---

## Q6. Why does L1 perform feature selection?

### Answer

When two features carry similar information, keeping both increases the L1 penalty without significantly improving prediction accuracy.

The optimizer therefore prefers keeping one feature while shrinking the other toward zero.

This naturally produces sparse models.

L1 does not explicitly identify correlated features—it simply minimizes the total loss.

---

## Q7. Why does the optimizer sometimes assign large weights to useless features without regularization?

### Answer

The optimizer only follows the loss function.

It cannot distinguish genuine relationships from accidental correlations.

If increasing the weight of a noisy feature reduces the training loss, even temporarily, the optimizer will increase that weight.

Regularization discourages this behavior by making large weights expensive.

---

## Q8. During optimization, aren't the weights constants? Why isn't the gradient of L1 zero?

### Answer

No.

During optimization, the weights are the variables being optimized.

The regularization loss is a function of the weights.

```text
L1 Loss

=

|w₁|

+

|w₂|

+

...

+

|wₙ|
```

When computing the gradient with respect to one weight, all other weights are treated as constants.

The optimizer computes how the loss changes if a weight changes slightly, which is why the regularization term contributes to every weight update.

---

# 💼 Section 3 — Scenario Questions

---

## Q9. Your training accuracy is 99%, but validation accuracy is only 81%. What does this indicate, and how can L1 help?

### Answer

The model is overfitting.

It has memorized the training data instead of learning generalizable patterns.

L1 Regularization encourages smaller weights and removes unnecessary features, reducing model complexity and improving generalization.

---

## Q10. Your dataset contains both "Years of Experience" and "Months of Experience." Which regularization technique would naturally tend to remove one of them?

### Answer

L1 Regularization.

These two features carry almost identical information.

Keeping both increases the L1 penalty without providing much additional predictive power.

The optimizer therefore tends to retain one feature and shrink the other toward zero.

---

## Q11. After adding a very large λ, both training and validation accuracy decrease. Why?

### Answer

The regularization penalty has become too strong.

The optimizer now prioritizes minimizing weight magnitudes over learning meaningful relationships.

Important weights also become too small, causing underfitting.

---

# 🚀 Section 4 — Senior / Staff Engineer Questions

---

## Q12. A teammate says:

> "L1 removes correlated features."

Would you agree?

### Answer

Partially.

L1 does not explicitly detect or remove correlated features.

It minimizes the total loss.

When multiple correlated features provide nearly identical predictive power, keeping all of them unnecessarily increases the L1 penalty.

The optimizer therefore often assigns weight to one feature while shrinking the others toward zero.

So feature selection is an emergent property of the optimization process, not an explicit rule.

---

## Q13. A teammate says:

> "Regularization simply prevents large weights."

Is that a complete explanation?

### Answer

No.

Large weights themselves are not the real problem.

The real problem is that very large weights often indicate the model is relying too heavily on specific features, increasing the risk of memorization and overfitting.

Regularization encourages the optimizer to use only the complexity that is truly necessary for good prediction.

---

## Q14. Explain L1 Regularization from an optimization perspective.

### Answer

Regularization modifies the optimization objective.

Instead of minimizing only Prediction Loss,

the optimizer minimizes

```text
Total Loss

=

Prediction Loss

+

λ × Regularization Loss
```

The weight update therefore depends on both

- Prediction Gradient
- Regularization Gradient

Every update becomes a balance between improving prediction accuracy and keeping the model simple.

This is why regularization integrates naturally into Gradient Descent rather than acting as a separate algorithm.

---

# ⭐ Staff Engineer Challenge

You are reviewing a fraud detection model.

Two models achieve almost identical validation accuracy.

### Model A

- Uses 180 features
- Most weights are large

### Model B

- Uses L1 Regularization
- Only 45 features remain active
- Validation accuracy is nearly identical

Which model would you deploy?

### Expected Discussion

Model B is generally the better engineering choice.

Reasons:

- Lower model complexity
- Better interpretability
- Lower risk of overfitting
- Faster inference
- Easier feature maintenance
- More robust to noisy inputs

However, deployment should not be based on sparsity alone.

A complete evaluation should also include:

- Cross-validation results
- Precision / Recall / F1 (especially for fraud detection)
- Feature stability over time
- Data drift analysis
- Latency and infrastructure constraints
- Business impact of false positives and false negatives