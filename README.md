# W1（11.30–12.06）PyTorch 内功：手写自动微分

## 本周目标

不看任何代码，从零写出一个能跑通的反向传播引擎，并用数值梯度验证它是对的。
**这是全部十周的地基。** 后面 attention 的推导、FSDP 的显存计算、kernel 的梯度，全建立在这一周上。

## 每日安排

| 天 | 任务 | 时间 |
|---|---|---|
| 一 | 装环境、跑通 `autograd_skeleton.py` 的正向；读 Karpathy《Yes you should understand backprop》 | 3h |
| 二 | 实现 `+`、`*`、`**` 的 backward，跑通 `test_gradcheck.py` 单变量部分 | 6h |
| 三 | 实现 `exp`、`log`、`relu`、`tanh`、`sigmoid`，补全 `backward()` 的拓扑排序 | 6h |
| 四 | 全部梯度校验通过（误差 < 1e-6）；用它训一个两层 MLP 拟合 sin(x) | 6h |
| 五 | 加 `matmul` 和广播；写 README；博客草稿 | 5h |
| 六/日 | 补落后 / 休息 | |

## 阅读清单

| 材料 | 读什么 | 花多久 |
|---|---|---|
| Karpathy《Yes you should understand backprop》 | 全篇，重点是「为什么数值梯度会骗你」 | 40 min |
| Karpathy `micrograd` 视频（前 1 小时） | 只看结构，**不要抄代码** | 1h |
| PyTorch Autograd 官方文档 "Autograd mechanics" | 计算图、叶子节点、`retain_grad` | 30 min |
| CS231n 反向传播笔记 | 计算图视角的链式法则 | 30 min |

## 验收标准（周五自查）

- [ ] `Value` 支持 `+ - * / ** exp log relu tanh matmul`
- [ ] `backward()` 用拓扑排序，同一个节点被多次使用时不重复累加错误
- [ ] 数值梯度校验：随机 100 组输入，相对误差 < 1e-6
- [ ] 两层 MLP 拟合 sin(x)，loss 降到 1e-3 以下
- [ ] README 里有一张计算图的手绘图或 mermaid 图

## 面试会问什么

- 为什么 `backward()` 必须先拓扑排序？不做会怎样？
- `x * x` 和 `x ** 2` 的 backward 有什么不同？为什么前者要对两次梯度求和？
- 广播（broadcast）的梯度怎么回传？为什么 `sum` 要沿广播维度做？
- 什么是「叶子节点」？为什么 `loss.backward()` 之后 `.grad` 会累积而不是覆盖？

## 常见坑

1. **先写 backward 再写 forward** —— 顺序反了，会导致你搞不清每个节点的输入输出。永远 forward 先行。
2. **拓扑排序用递归会爆栈** —— 深图（>1000 层）会 RecursionError，用显式栈 + visited 集合。
3. **忘了 `zero_grad()`** —— 梯度会一直累加，看着 loss 不降，排查半天。
4. **数值梯度用 `h=1e-4` 而不是 `1e-7`** —— float32 下太小的 h 会被舍入误差吃掉，用 1e-4 到 1e-5。

## 交付物

```
W01-autograd/
├─ autograd_skeleton.py   ← 你的实现（改这个文件）
├─ test_gradcheck.py      ← 验收测试，全绿才算过
└─ README.md              ← 加一段「我踩的坑」
```
