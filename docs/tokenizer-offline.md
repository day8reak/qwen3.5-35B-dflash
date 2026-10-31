# tokenizer 离线准备

[文档首页](README.md) · [下一步：文字＋预读运行](board-text-input-output.md)

DDK服务器可以全程离线。`prepare_board_tokenizer.py --source-dir`只读取本地文件、检查大小和SHA256、生成板端词表；不会下载文件。文本请求也直接从本地词表编码，不需要连接Hugging Face。

## 1. 在能联网的电脑准备三个文件

使用固定版本 `Qwen/Qwen3.5-35B-A3B`，revision为 `59d61f3ce65a6d9863b86d2e96597125219dc754`，总计约12.8MB。可以用浏览器分别下载：

| 文件 | 下载地址 | 字节数 |
|---|---|---:|
| tokenizer.json | [固定版本下载](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/59d61f3ce65a6d9863b86d2e96597125219dc754/tokenizer.json?download=true) | 12807982 |
| tokenizer_config.json | [固定版本下载](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/59d61f3ce65a6d9863b86d2e96597125219dc754/tokenizer_config.json?download=true) | 16710 |
| chat_template.jinja | [固定版本下载](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/59d61f3ce65a6d9863b86d2e96597125219dc754/chat_template.jinja?download=true) | 7756 |

已有这个准确版本的本地模型目录时，直接复制这三个文件即可。不要取网页HTML、Git LFS指针或不同版本的文件；身份由仓库的[锁定清单](../specs/tokenizer.lock.json)检查。

联网电脑也可运行仓库现有准备脚本，随后只把输出的`source/`目录传到服务器：

```bash
python3 targets/cpro/scripts/prepare_board_tokenizer.py \
  --output /实际路径/tokenizer-transfer
```

这个准备脚本仅需Python标准库；不会下载模型权重。不要在证书不可用的服务器运行省略`--source-dir`的下载模式，也无需关闭TLS证书检查。

## 2. 传到DDK服务器并离线校验

推荐最终目录结构：

```text
$AI_RUN_DIR/tokenizer-offline/
  source/
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
```

若收到本次提供的`qwen35-tokenizer-offline.tar.gz`，包内已经有`source/`和`SHA256SUMS`。先将压缩包传到服务器，选择一个尚不存在的解压目录：

本次包大小为3,521,447字节，SHA256为`68e5346d7072a6aba9e684678f74c2626053cdd2a080ac3adefa4f3de6b28b6e`；不包含模型权重或Python依赖。

```bash
mkdir "$AI_RUN_DIR/tokenizer-offline"
tar -xzf /实际路径/qwen35-tokenizer-offline.tar.gz \
  -C "$AI_RUN_DIR/tokenizer-offline"
(cd "$AI_RUN_DIR/tokenizer-offline" && sha256sum -c SHA256SUMS)
```

用浏览器单独下载并传输时，把三个文件直接放入上面的`source/`；下面脚本同样会做完整SHA校验，无需自行创建`SHA256SUMS`。

```bash
export TOKENIZER_DIR="$AI_RUN_DIR/tokenizer-offline/source"

python3 "$REPO_DIR/targets/cpro/scripts/prepare_board_tokenizer.py" \
  --source-dir "$TOKENIZER_DIR" \
  --output "$AI_RUN_DIR/tokenizer-offline/prepared"
```

预期输出为`BOARD_TOKENIZER_PREPARED`。保留三个源文件；运行参数`--tokenizer-dir`指向它们所在的`source/`，不是`prepared/`或`tokenizer.bin`。

已准备成功时直接复用目录，不重跑到已有输出目录。若已有词表位于其他地方，只需将`TOKENIZER_DIR`改成真实目录；没有强制移动或重新下载要求。

## 3. 检查主机Python依赖

生成词表使用标准库；文字请求的主机准入及最终文本对照需要`tokenizers`。板端不需要Python或该包。先在实际运行请求的`custom_pypto`环境检查：

```bash
python3 -c 'import tokenizers; print(tokenizers.__version__)'
```

能导入时使用现有环境。本项目主机对照使用`tokenizers==0.22.2`。如果缺包，在**Python版本、操作系统和CPU架构相同**的联网环境下载wheel及依赖：

```bash
python3 -m pip download --only-binary=:all: \
  --dest tokenizer-wheelhouse 'tokenizers==0.22.2'
```

把整个wheelhouse传到服务器后，在`custom_pypto`环境离线安装：

```bash
python3 -m pip install --no-index \
  --find-links /实际路径/tokenizer-wheelhouse 'tokenizers==0.22.2'
```

联网电脑若是Windows或其他Python版本，不要直接复制它默认下载的wheel。可先在服务器用下面命令确认目标，再用相同Linux环境下载，或用pip的目标平台参数下载匹配wheel：

```bash
python3 -c 'import sys, platform; print(sys.version); print(platform.system(), platform.machine())'
```

## 4. 同一次请求启用文字和预读

继续执行[文字＋预读完整命令](board-text-input-output.md#2-运行文字版完整闭环)。其中使用：

```bash
--tokenizer-dir "$TOKENIZER_DIR" \
--text '你好，请介绍一下自己。' \
--resident --draft-preread
```

三个源文件、板端词表和本地Python依赖准备好后，服务器无需互联网。服务器到板端的HDC仍用于一次下发与最终回收；生成期间不逐token与服务器交互。

本地离线准备已在禁止下载函数与socket连接的测试中通过，记录见[离线验证](../specs/board-tokenizer-offline-validation.json)。这项检查不代表文字请求已经在用户板端执行。
