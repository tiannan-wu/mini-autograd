# mini-autograd

A ~120-line scalar automatic differentiation engine, written from scratch to
understand what `loss.backward()` actually does.

> 中文笔记 / Chinese notes: [docs/notes-zh.md](docs/notes-zh.md)

Supports `+  -  *  /  **  exp  log  relu  tanh`. Each operation creates a `Value`
node holding four things: `data` (the number), `grad` (accumulated gradient),
`_prev` (its inputs), and a closure `_backward` that knows how to route the
gradient back to those inputs. Reverse-mode autodiff runs a topological sort over
the graph and executes the closures in reverse order.

## Results

Gradients are verified against central-difference numerical gradients
(6 expressions, 11 assertions, `atol=1e-10`, `rtol=1e-6`):

| test | relative error |
|---|---|
| composite (tanh / pow / log) | 1.7e-12 – 3.2e-11 |
| shared node (`x*x + x*y`) | 7.5e-12 – 3.7e-11 |
| relu | 2.1e-12 – 5.6e-12 |
| div / sub | 3.2e-12 – 1.9e-11 |
| exp (sigmoid) | 2.5e-11 |
| deep chain (50 × tanh) | 2.5e-3 * |

\* the 50-layer gradient is only 3.5e-9, so the difference quotient loses
precision; absolute error is 8.6e-12. This limit comes from the *check*, not from
the engine: compared against 60-digit `mpmath`, the engine's own relative error
stays around 1e-15 at every depth.

## Usage

```python
from autograd import Value

a, b = Value(2.0), Value(-3.0)
y = (a * b + Value(10.0)).tanh()
y.backward()

print(y.data)    # forward value
print(a.grad)    # dL/da
```

## Layout

```
autograd.py          the engine (~120 lines)
test_gradcheck.py    numerical gradient checks
```

## What I got stuck on

1. **Wrapping a constant into `_prev`.** `exp` takes one input; `math.e` is not a
   graph node. Same for the exponent in `x ** n`.
2. **Assigning `out.grad` inside the forward pass.** Upstream gradient
   (`out.grad`) and local derivative are different quantities — only the local
   derivative belongs in `_backward`.
3. **Putting `topo.append(child)` inside the `for` loop.** The root node never
   entered the topological order, so its gradient never propagated.
4. **`if self._prev is None`.** `_prev` is `set()`, never `None`. The base case
   has to be an empty set.
5. **Writing `self.grad` on the right-hand side of the assignment** (twice). The
   accumulator is a receiver, not an operand; `self.grad *= ...` silently zeroes
   the gradient.

## Open question

Why does PyTorch's autograd engine use a ready queue (a node is scheduled once
all of its consumers are done) instead of precomputing a topological order?
My guess is parallelism — but I would like to see the actual constraint.

## Reference

The overall structure follows Karpathy's micrograd; the implementation, the test
harness, and these notes are mine.
