"""
W1 骨架：从零实现一个自动微分引擎（对标 micrograd，但要自己写）

目标：让 test_gradcheck.py 全绿。
规则：不许 import torch / numpy 的自动微分，只准用 math 和 list。

跑法：
    python autograd_skeleton.py     # 看到 2.0 就算正向通了
    python test_gradcheck.py        # 全绿才算过关
"""

from __future__ import annotations

import math

from torch.autograd import backward


class Value:
    """标量自动微分节点。

    每个 Value 记住三件事：
      1. data    —— 前向数值
      2. grad    —— 损失对本节点的梯度（初始 0.0）
      3. _prev / _backward —— 计算图里的父节点，以及「怎么把梯度传回去」
    """

    def __init__(self, data: float, _children: tuple["Value", ...] = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ---------- 基础运算（已经给你写好了两个，照着写） ----------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # 加法把梯度原样分给两边
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), "-")

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += out.grad * other.data**(-1)
            other.grad -=  out.grad * out.data * other.data**(-1)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO: 补全。提示：乘法是「交叉相乘」，d(a*b)/da = b
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    # ---------- 需要你实现的运算 ----------

    def __pow__(self, other: float):
        other = other if isinstance(other, Value) else Value(other)
        # TODO: 只有标量指数，d(x^n)/dx = n * x^(n-1)
        out = Value(self.data ** other.data, (self, other), "**")

        def _backward():
            self.grad += (other.data * self.data**(other.data-1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        # TODO: d(e^x)/dx = e^x，可以复用 out.data
        out = Value(math.exp(self.data), (self,), "exp**x")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        # TODO: d(ln x)/dx = 1/x
        out = Value(math.log(self.data), (self,), "ln(x)")

        def _backward():
            self.grad += (1/self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        # TODO: x>0 时梯度原样，否则 0
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # TODO: d(tanh x)/dx = 1 - tanh(x)^2
        out = Value(math.tanh(self.data), (self,), "tanh(x)")
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other


    # ---------- 核心：反向传播 ----------

    def backward(self):
        topo, visited = [], set()
        stack = [(self, False)]  # (节点, 是否已展开)

        while stack:
            node, expanded = stack.pop()

            if expanded:  # 第二次出栈 → 子节点都处理完了
                topo.append(node)
                continue

            if node in visited:
                continue

            visited.add(node)

            stack.append((node, True))  # 先压"收尾"
            for child in node._prev:  # 再压子节点（后进先出）
                if child not in visited:
                    stack.append((child, False))

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()





    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# --------------------------- 自测 ---------------------------
if __name__ == "__main__":
    # 1) 基础的加法与乘法
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b + c          # 2*-3 + 10 = 4
    print("正向:", e.data)  # 期望 4.0


    # 2) 复合表达式
    d = a * b
    f = d + c
    f.backward()
    print("dL/da =", a.grad, "(期望 -3.0)")
    print("dL/db =", b.grad, "(期望 2.0)")
    print("dL/dc =", c.grad, "(期望 1.0)")

    # 3) 同一个节点被用两次：x*x 的梯度应该是 2x
    x = Value(3.0)
    y = x * x
    y.backward()
    print("d(x^2)/dx =", x.grad, "(期望 6.0，如果得到 3.0 说明梯度被覆盖了)")
