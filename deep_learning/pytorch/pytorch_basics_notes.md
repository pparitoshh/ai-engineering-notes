# ML Notes: PyTorch Basics — nn.Module, Parameters & Training Loop

**Last Updated**: 2026-05-31
**Folder**: `deep_learning/pytorch/`

---

## 1. Core Concept

PyTorch makes every step of training explicit — forward pass, loss computation, gradient calculation, and weight update are all written manually in a loop. This contrasts with sklearn where `model.fit()` hides all of this internally.

---

## 2. sklearn → PyTorch Mental Model

| sklearn | PyTorch equivalent |
|---|---|
| `model.fit(X_train, y_train)` | the entire `for epoch in range(N):` loop |
| `model.predict(X_test)` | `model.eval()` + `torch.inference_mode()` + `model(X_test)` |
| loss is hidden | `loss_fn(y_preds, y_train)` — you define and compute it |
| optimizer is hidden | `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` |

---

## 3. `nn.Module` — Base Class for All Models

```python
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
```

- Every PyTorch model inherits from `nn.Module`
- `super().__init__()` initializes the machinery: parameter tracking, save/load, train/eval modes
- **Forgetting `super().__init__()` breaks everything** ⚠️

**What `nn.Module` gives you:**
- `.parameters()` — lists all learnable weights
- `.train()` / `.eval()` — mode switching
- `.state_dict()` — save/load weights

---

## 4. `nn.Parameter` — Marking a Tensor as Learnable

```python
self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
self.bias    = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
```

- `torch.randn(1)` — random starting value (random initialization)
- `nn.Parameter(...)` — tells `nn.Module`: *"track this, update this during training"*
- Without `nn.Parameter`, the optimizer **will not see the tensor** and weights never update ⚠️
- `nn.Parameter` sets `requires_grad=True` by default — writing it explicitly is redundant but readable

---

## 5. `requires_grad=True` — Autograd

- Tells PyTorch to track all operations on this tensor
- When `loss.backward()` is called, PyTorch computes `d(loss)/d(weight)` for every tensor with `requires_grad=True`
- This is PyTorch's automatic differentiation (autograd) system

---

## 6. `forward()` — The Prediction Equation

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * self.weights + self.bias
```

- Defines `y = wx + b` — the linear regression equation
- Called automatically when you do `model(X_train)` — never call `forward()` directly
- In sklearn this is hidden inside `LinearRegression.predict()`

---

## 7. The Training Loop — Line by Line

```python
model_0.train()           # Set training mode (affects Dropout, BatchNorm — not this model)
y_preds = model_0(X_train)  # Forward pass — calls forward()
loss = loss_fn(y_preds, y_train)  # Compute loss (e.g. MAE)
optimizer.zero_grad()     # CLEAR gradients from last step — they accumulate by default ⚠️
loss.backward()           # Backprop — compute gradients
optimizer.step()          # Update weights using gradients
```

**Why `zero_grad()`?** PyTorch accumulates gradients across steps by default. Without zeroing, epoch 2 gradients add to epoch 1 gradients → weights explode. sklearn resets internally and never exposes this.

---

## 8. Inference / Evaluation Block

```python
model_0.eval()
with torch.inference_mode():
    test_pred = model_0(X_test)
```

- `model.eval()` — switches off Dropout, freezes BatchNorm stats
- `torch.inference_mode()` — disables computation graph building → saves memory and compute
- Always wrap predictions in this block when not training

---

## 9. `model_0.train()` — What It Actually Does

⚠️ **Common confusion (from sklearn background):** This does NOT train the model. It just sets a mode flag.

- Affects layers like `Dropout` (active in train, off in eval) and `BatchNorm` (uses batch stats in train, running stats in eval)
- For simple models with no such layers, it's a no-op — but always write it as best practice

---

## 10. `nn.Linear` vs Manual Parameters

Your `SimpleLinearRegression` manually implements what `nn.Linear(1, 1)` does internally:
- `nn.Linear` already has `weight` and `bias` as `nn.Parameter` under the hood
- Building it manually is a learning exercise to understand the internals
- Real models use `nn.Linear`, `nn.Conv2d`, etc.

---

## 11. Key Tensor Concepts

| Concept | What it is |
|---|---|
| `torch.randn(1)` | Random tensor of shape `(1,)` |
| `nn.Parameter` | Tensor registered as learnable weight |
| `requires_grad=True` | Enable gradient tracking via autograd |
| `loss.backward()` | Compute all gradients via backprop |
| `optimizer.step()` | Apply gradients to update weights |
| `detach().numpy()` | Detach from computation graph before converting to numpy |

---

## 12. FAQ Bank

### L1 — Foundational
Q: What is `nn.Module`?
A: The base class for all PyTorch models. It tracks parameters, enables save/load, and provides train/eval mode switching. Every custom model inherits from it.

Q: What does `model.train()` do?
A: Sets the model to training mode — affects Dropout and BatchNorm layers. Does NOT perform training. Always called before the training loop.

### L2 — Intermediate
Q: Why do we call `optimizer.zero_grad()`?
A: PyTorch accumulates gradients by default. Without zeroing, gradients from previous steps add up, causing incorrect weight updates and potential divergence.

Q: What is `torch.inference_mode()`?
A: A context manager that disables gradient computation graph building during prediction. Saves memory and compute — use it whenever you're not training.

### L3 — Advanced
Q: What's the difference between `nn.Parameter` and a regular tensor?
A: `nn.Parameter` is registered with the parent `nn.Module`, so it appears in `.parameters()` and gets updated by the optimizer. A plain tensor assigned as `self.x = torch.randn(1)` is invisible to the optimizer.

Q: Why does `requires_grad=True` inside `nn.Parameter` seem redundant?
A: It is. `nn.Parameter` sets `requires_grad=True` by default. Writing it explicitly makes the intent visible — useful when learning the internals.

---

## 13. Resources

| Type | Resource | URL |
|---|---|---|
| Official | PyTorch nn.Module docs | https://pytorch.org/docs/stable/generated/torch.nn.Module.html |
| Official | Autograd tutorial | https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html |
| Course | Zero to Mastery PyTorch | https://www.learnpytorch.io/ |
| Book | Deep Learning with PyTorch | https://pytorch.org/deep-learning-with-pytorch |
