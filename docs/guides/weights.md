# 权重准备

[首页](../README.md) · [环境配置](environment-setup.md)

## 权重来源

DFlash 官方代码：[z-lab/dflash（GitHub）](https://github.com/z-lab/dflash)。目标模型和配套草稿权重托管在 Hugging Face，以下下载链接固定到本工程锁定的版本：

| 用途 | 权重下载 | 锁定版本 |
|---|---|---|
| 目标模型 | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/59d61f3ce65a6d9863b86d2e96597125219dc754) | `59d61f3` |
| DFlash 草稿模型 | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash/tree/52cb554b4995dede3e2e1bdb129cdb1f3529332b) | `52cb554` |

完整 revision 和已核对配置/索引的哈希见 [upstream.lock.json](../../specs/upstream.lock.json)。完整权重需另行下载，本仓库不包含权重或量化成品；当前开发环境仅下载并核对了配置/索引，完整 35B 权重验证仍待完成。

### 在服务器下载

在能访问 Hugging Face 及模型文件下载服务的服务器执行。没有 `hf` 命令时，使用下面的 Python 方式即可。两种方式都使用 `huggingface_hub`；当前 Python 环境已安装该包时跳过安装：

```bash
python3 -m pip install 'huggingface_hub>=1.5,<2'
```

沿用本次任务的 `AI_RUN_DIR`。未设置时，把下面的 `/your/run/storage/qwen35-cpro-001` 换成实际的仓库外目录；工作区用户直接使用 `ws session start` 分配的目录。

```bash
export AI_RUN_DIR="${AI_RUN_DIR:-/your/run/storage/qwen35-cpro-001}"
mkdir -p "$AI_RUN_DIR/weights"

export QWEN35_TARGET_WEIGHTS="$AI_RUN_DIR/weights/Qwen3.5-35B-A3B"
export QWEN35_DRAFT_WEIGHTS="$AI_RUN_DIR/weights/Qwen3.5-35B-A3B-DFlash"
```

下面两种下载方式任选一种，均固定到相同版本。

**方式一：Python 下载，不依赖 `hf` 命令。** 在设置好上述路径的同一终端执行：

```bash
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-35B-A3B",
    revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
    local_dir=os.environ["QWEN35_TARGET_WEIGHTS"],
)

snapshot_download(
    repo_id="z-lab/Qwen3.5-35B-A3B-DFlash",
    revision="52cb554b4995dede3e2e1bdb129cdb1f3529332b",
    local_dir=os.environ["QWEN35_DRAFT_WEIGHTS"],
)
PY
```

**方式二：已有 `hf` 命令时使用 CLI。**

```bash
# 目标模型：权重、配置和 tokenizer
hf download Qwen/Qwen3.5-35B-A3B \
  --revision 59d61f3ce65a6d9863b86d2e96597125219dc754 \
  --local-dir "$QWEN35_TARGET_WEIGHTS"

# 配套 DFlash 草稿模型
hf download z-lab/Qwen3.5-35B-A3B-DFlash \
  --revision 52cb554b4995dede3e2e1bdb129cdb1f3529332b \
  --local-dir "$QWEN35_DRAFT_WEIGHTS"
```

这里下载的是原始权重，后续在服务器量化。目标权重约 72 GB，下载目录建议至少留 100 GB 磁盘空间；校准、AIR、OMC 还需额外空间。下载中断后可重跑同一条命令或整段 Python，已完成且未变化的文件会复用，保留下载目录中的 `.cache/huggingface/` 元数据。参数与下载机制见 [Python 下载文档](https://huggingface.co/docs/huggingface_hub/guides/download#download-an-entire-repository)和 [CLI 文档](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download)。

下载完成后，当前 shell 已设置好后续流程需要的两个权重路径。新开终端时重新设置它们，指向实际下载目录。


量化与参考命令见 [PyTorch 执行指南](pytorch-guide.md)。直接 GE 当前只接受未量化 FP16 prepared；完整模型 FP16 权重不是板端低内存部署方案。
