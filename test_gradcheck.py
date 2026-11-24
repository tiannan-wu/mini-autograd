"""
W1 验收：数值梯度校验（改进版，防止"悄悄放水"）

原版用的是：
    denom = max(1.0, |ana|, |n|)
    rel = |ana - n| / denom
梯度很小时 denom 退化成 1.0，相对误差悄悄变成绝对误差——看起来漂亮，实际是放水。

这版用 allclose 的标准形式：
    |ana - num| <= atol + rtol * |num|
      - 大梯度：主要看 rtol（相对）
      - 小梯度：主要看 atol（绝对）

阈值怎么定的（不是拍脑袋）：
    rtol = 1e-6    中心差分 h=1e-5 时噪声约 1e-10，留 4 个数量级余量
    atol = 1e-10   50 层深链的梯度只有 3.5e-9，实测噪声 8.55e-12，留 1 个数量级余量

注意 "深链 50 层" 那一项的相对误差约 2.5e-3（0.25%），这是正常的：
梯度本身只有 3.5e-9，而 f(x±h) 的值在 1 附近，相减时丢失了一位有效数字。
想更准只能换更高精度的参考（mpmath），不是你的引擎有问题。

跑法：python test_gradcheck.py
"""

import random

from autograd import Value

random.seed(42)

H = 1e-5
ATOL = 1e-10
RTOL = 1e-6


def numeric_grad(f, args, idx, h=H):
    """中心差分：dL/dx ≈ (f(x+h) - f(x-h)) / 2h"""
    lo = list(args)
    hi = list(args)
    lo[idx] -= h
    hi[idx] += h
    return (f(*hi) - f(*lo)) / (2 * h)


def check(name, build, args):
    vals = [Value(a) for a in args]
    out = build(*vals)
    out.backward()

    numeric = [
        numeric_grad(lambda *xs: build(*[Value(x) for x in xs]).data, args, i)
        for i in range(len(args))
    ]

    ok = True
    for i, n in enumerate(numeric):
        ana = vals[i].grad
        err = abs(ana - n)
        tol = ATOL + RTOL * abs(n)
        good = err <= tol
        ok &= good
        rel = err / abs(n) if n != 0 else float("inf")
        print(f"  [{'OK  ' if good else 'FAIL'}] {name} d/dx{i}: "
              f"ana={ana:+.6e} num={n:+.6e} "
              f"|diff|={err:.2e} tol={tol:.2e} rel={rel:.1e}")
    return ok


def main():
    results = []

    # --- 1. 复合表达式：tanh + 幂 + log ---
    def f1(a, b, c):
        return (a * b + c).tanh() + (a ** 2) * c.log()
    results.append(check("composite", f1, [0.7, -0.3, 1.4]))

    # --- 2. 同一个节点用两次（梯度必须累加，不能覆盖） ---
    def f2(x, y):
        return x * x + x * y          # d/dx = 2x+y, d/dy = x
    results.append(check("reuse-node", f2, [1.3, 2.1]))

    # --- 3. relu 的分段 ---
    def f3(a, b):
        return (a * b).relu() * a
    results.append(check("relu", f3, [1.2, 0.8]))

    # --- 4. 深链：考验拓扑排序 + 梯度长程传递 ---
    def f4(x):
        v = x
        for _ in range(50):
            v = (v * 0.9 + 0.1).tanh()
        return v
    results.append(check("deep-50", f4, [0.4]))

    # --- 5. 除法与减法：走 __truediv__ → __mul__ + __pow__ 的组合路径 ---
    def f5(x, y):
        return x / (y + 2.0) - x * y
    results.append(check("div-sub", f5, [1.3, 2.1]))

    # --- 6. 指数：exp 的导数复用 out.data ---
    def f6(x):
        return x.exp() / (1.0 + x.exp())      # sigmoid(x)
    results.append(check("exp-sigmoid", f6, [0.5]))

    print()
    if all(results):
        print(f"[PASS] W1 PASSED  ({len(results)} groups, atol={ATOL:g}, rtol={RTOL:g})")
        return 0
    print("[FAIL] some groups failed -- check the corresponding _backward")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
