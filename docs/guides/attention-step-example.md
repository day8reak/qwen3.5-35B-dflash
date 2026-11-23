# 单测 L0 的 `F.silu(conv_out)`

两台服务器首次使用时，在仓库根目录执行 `git pull --ff-only origin main`。沿用原来配置好的 310P、DDK 和 HDC 环境变量，脚本会选择对应的 Conda 环境。

**78 · 准备一次。** 从 L0 的真实 forward 采集卷积输出 `conv_out`，保存同输入的 CPU、310P 参考结果，并截取 SiLU 的 AIR。

```bash
bash targets/cpro/scripts/conv_silu.sh prepare
```

**35 · 每测一次执行一遍。** 脚本拉取 78 的同一份输入，重新编译 SiLU OMC、上板执行，并打印 valid19 和 token18 的 CPU、310P、CPro 三端矩阵。

```bash
bash targets/cpro/scripts/conv_silu.sh board
```

每次 `board` 都使用新目录，完整指标保存在终端打印的 `matrices.csv`。上次运行中断时，可把打印的运行目录追加到 `board` 后面续跑。重复执行 `prepare` 会校验并复用已准备的数据。

源 attention AIR 放在自定义位置时，在 78 设置 `ATTENTION_AIR` 为含 `manifest.json` 的原始批次目录。默认源 AIR 已删除时，脚本先恢复用于截取的原 attention 批次。两机的运行根目录不同时，在 35 设置 `AIR_ON_78` 为 78 本次准备的 SiLU `air/` 绝对路径。
