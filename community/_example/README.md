# 示例投稿

这里的两个文件夹只用来演示投稿的格式，不参加社区榜，也不是谁的真实投稿。

| 文件夹 | 说明 |
|---|---|
| `ctf/chen-006__stand-still/` | bot 类投稿的样子：`bot.cpp` 加 `meta.json`。代码是第 3 集选手包里的模板，三个角色原地不动 |
| `warehouse/chen-006__kimi-k3-round1/` | 仓储调度投稿的样子：`src/` 加 `meta.json`。代码是 Kimi K3 第 1 轮交的原样 |

可以拿它们试一下自测脚本：

```bash
python community/check.py community/_example/ctf/chen-006__stand-still
python community/check.py community/_example/warehouse/chen-006__kimi-k3-round1
```

真正投稿时，文件夹要放在 `community/<集>/` 下，不是 `community/_example/<集>/` 下。规则见 [`../README.md`](../README.md)。
