# 环境配置

复用已安装的custom_pypto、目标DDK、OHOS Native SDK和HDC。命令在DDK服务器执行，替换示例路径；已有项目配置和词表可直接复用。源码由联网电脑同步后传入，保留仓库中的third_party目录，构建无需联网。

## 1. 恢复工具链

```bash
conda activate custom_pypto
source '/实际DDK根目录/tools/tools_ascendc/set_ascendc_env.sh'
export REPO_DIR='/实际路径/qwen3.5-35B-dflash'
```

## 2. 加载项目配置

仅首次需要填下面的路径与设备值。已有仓库旁qwen35-cpro.env时保留它。

```bash
if ! test -f "$REPO_DIR/../qwen35-cpro.env"; then
  export AI_RUN_DIR="${REPO_DIR}-run"
  export CANN_ROOT='/实际CANN版本目录'
  export CPRO_DDK_ID='100.520.020.010'
  export OHOS_NATIVE_SDK='/实际SDK目录/openharmony/native'
  export BIP='实际HDC服务器IP:端口'
  export SN='实际设备序列号'
  # 首次创建配置会提示再次加载；下一行source负责加载。
  source "$REPO_DIR/targets/cpro/scripts/env.sh"
fi
source "$REPO_DIR/targets/cpro/scripts/env.sh"
```

## 3. 离线准备词表

在可联网电脑下载[离线词表包（3.52 MB）](https://raw.githubusercontent.com/day8reak/qwen3.5-35B-dflash/main/docs/assets/qwen35-tokenizer-offline.tar.gz)和[SHA256清单](https://raw.githubusercontent.com/day8reak/qwen3.5-35B-dflash/main/docs/assets/qwen35-tokenizer-offline.tar.gz.sha256)，再传到服务器。包内是锁定版本的三个原始词表文件，不含权重或Python依赖。已有source和prepared目录时跳过解压／准备，只设置TOKENIZER_DIR。

```bash
mkdir "$AI_RUN_DIR/tokenizer-offline"
tar -xzf /实际路径/qwen35-tokenizer-offline.tar.gz \
  -C "$AI_RUN_DIR/tokenizer-offline"
(cd "$AI_RUN_DIR/tokenizer-offline" && sha256sum -c SHA256SUMS)

export TOKENIZER_DIR="$AI_RUN_DIR/tokenizer-offline/source"
python3 "$REPO_DIR/targets/cpro/scripts/prepare_board_tokenizer.py" \
  --source-dir "$TOKENIZER_DIR" \
  --output "$AI_RUN_DIR/tokenizer-offline/prepared"

python3 -c 'import numpy, tokenizers; print(tokenizers.__version__)'
cmake --version
```

## 4. 仅缺少tokenizers时

在与服务器相同Python版本、OS和CPU架构的联网环境下载wheel，整个目录传入服务器。不要复制Windows环境默认下载的wheel到Linux。

```bash
# 联网电脑
python3 -m pip download --only-binary=:all: \
  --dest tokenizer-wheelhouse 'tokenizers==0.22.2'

# DDK服务器的custom_pypto环境
python3 -m pip install --no-index \
  --find-links /实际路径/tokenizer-wheelhouse 'tokenizers==0.22.2'
```
