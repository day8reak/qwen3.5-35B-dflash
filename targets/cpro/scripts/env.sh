#!/usr/bin/env bash
# Source this file from Bash. The private config lives beside the repository.
if [[ -z ${BASH_VERSION:-} ]]; then
  printf '%s\n' 'cpro-env: use Bash to source this file' >&2
  return 1
fi
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  printf '%s\n' 'Usage: source targets/cpro/scripts/env.sh [private-config-file]' >&2
  exit 1
fi

_qwen35_load_env() (
  # A subshell prevents an incomplete config from changing the caller.
  repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P) || return 1
  config=${1:-${QWEN35_ENV_CONFIG:-"$repo/../qwen35-cpro.env"}}
  config=$(realpath -m -- "$config") || return 1
  required=(REPO_DIR AI_RUN_DIR MODEL_PYTHON CANN_ROOT CPRO_DDK_ROOT CPRO_DDK_ID
            OHOS_NATIVE_SDK HIAI_INCLUDE_DIR HIAI_LIBRARIES HDC BIP SN)
  optional=(BOARD_HIAI_LIB_DIR QWEN35_TARGET_WEIGHTS QWEN35_DRAFT_WEIGHTS)
  if [[ ! -f $config ]]; then
    # Capture declared values when available; never invent a device or DDK ID.
    REPO_DIR=$repo
    AI_RUN_DIR=${AI_RUN_DIR:-"$repo-run"}
    MODEL_PYTHON=${MODEL_PYTHON:-${CONDA_PREFIX:+"$CONDA_PREFIX/bin/python3"}}
    CPRO_DDK_ROOT=${CPRO_DDK_ROOT:-${DDK_PATH:-}}
    HDC=${HDC:-$(command -v hdc || true)}
    OHOS_NATIVE_SDK=${OHOS_NATIVE_SDK:-${OHOS_SDK:-}}
    HIAI_INCLUDE_DIR=${HIAI_INCLUDE_DIR:-auto}
    HIAI_LIBRARIES=${HIAI_LIBRARIES:-auto}
    if [[ $config == "$repo/"* || $config == "$(realpath -m -- "$AI_RUN_DIR")/"* ]]; then
      printf '%s\n' 'cpro-env: keep the private config outside the repository and AI_RUN_DIR' >&2
      return 1
    fi
    mkdir -p -- "$(dirname -- "$config")" || return 1
    (
      umask 077
      set -o noclobber
      {
        printf '%s\n' '# Private Bash config: fill empty required values once, then source env.sh.'
        printf '%s\n' '# BIP is the HDC SERVER IP:port; SN is the device serial.'
        printf '%s\n' '# HIAI_LIBRARIES: host ARM64 library files, separated by colons.'
        printf '%s\n' '# HIAI paths may be auto: unique header / AArch64 libhiai.so in an ohos directory within this DDK.'
        for name in "${required[@]}"; do printf '%s=%q\n' "$name" "${!name-}"; done
      } > "$config"
    ) || return 1
    printf 'cpro-env: created %s\nFill its required values, then source this script again.\n' "$config" >&2
    return 1
  fi
  # The config is a trusted, user-owned Bash file. Its output is not shell code.
  for name in "${required[@]}" "${optional[@]}"; do unset "$name"; done
  source "$config" >&2 || return 1
  errors=0
  for name in "${required[@]}"; do
    if [[ -z ${!name-} ]]; then printf 'cpro-env: missing %s in %s\n' "$name" "$config" >&2; errors=1; fi
  done
  [[ $errors == 0 ]] || return 1
  if [[ $HIAI_INCLUDE_DIR == auto || $HIAI_LIBRARIES == auto ]]; then
    if [[ $CPRO_DDK_ROOT != /* || ! -d $CPRO_DDK_ROOT ]]; then
      printf '%s\n' 'cpro-env: HIAI auto lookup requires the selected absolute CPRO_DDK_ROOT' >&2
      return 1
    fi
    if [[ $HIAI_LIBRARIES == auto ]] && ! command -v readelf >/dev/null 2>&1; then
      printf '%s\n' 'cpro-env: readelf is required for HIAI_LIBRARIES=auto; alternatively set the exact ARM64 library path' >&2
      return 1
    fi
    headers=() hiai_libs=() candidates=()
    # Read filenames only, confined to the selected DDK, depth <= 8. Do not
    # follow directory symlinks into another SDK or import vendor code.
    while IFS= read -r -d '' path; do
      [[ -f $path ]] || continue
      resolved=$(realpath -- "$path") || continue
      [[ $resolved == "$(realpath -- "$CPRO_DDK_ROOT")/"* ]] || continue
      if [[ ${path##*/} == HiAiModelManagerService.h ]]; then
        headers+=("$path")
      else
        candidates+=("$path")
        [[ ${path,,} == */ohos/* || ${path,,} == */*-ohos/* ]] || continue
        elf=$(LC_ALL=C readelf -h -- "$path" 2>/dev/null) || continue
        if [[ $elf == *ELF64* && $elf == *AArch64* && $elf == *DYN* && $elf == *"little endian"* ]]; then
          hiai_libs+=("$path")
        fi
      fi
    done < <(
      if command -v rg >/dev/null 2>&1; then
        rg --files --hidden --no-ignore --max-depth 8 -0 \
          -g HiAiModelManagerService.h -g libhiai.so "$CPRO_DDK_ROOT"
      else
        find "$CPRO_DDK_ROOT" -maxdepth 8 \( -name HiAiModelManagerService.h -o -name libhiai.so \) -print0
      fi
    )
    if [[ $HIAI_INCLUDE_DIR == auto ]]; then
      if [[ ${#headers[@]} == 1 ]]; then HIAI_INCLUDE_DIR=${headers[0]%/*}; else
        printf 'cpro-env: found %s matching headers; set HIAI_INCLUDE_DIR explicitly\n' "${#headers[@]}" >&2
        printf '  %s\n' "${headers[@]:0:12}" >&2
        return 1
      fi
    fi
    if [[ $HIAI_LIBRARIES == auto ]]; then
      if [[ ${#hiai_libs[@]} == 1 ]]; then HIAI_LIBRARIES=${hiai_libs[0]}; else
        printf 'cpro-env: found %s AArch64 OHOS libhiai.so candidates; set HIAI_LIBRARIES explicitly\n' "${#hiai_libs[@]}" >&2
        printf '  %s\n' "${candidates[@]:0:12}" >&2
        return 1
      fi
    fi
    printf 'cpro-env: HIAI include: %s\n  link libraries: %s\n' "$HIAI_INCLUDE_DIR" "$HIAI_LIBRARIES" >&2
  fi
  for name in REPO_DIR AI_RUN_DIR MODEL_PYTHON CANN_ROOT CPRO_DDK_ROOT OHOS_NATIVE_SDK HIAI_INCLUDE_DIR; do
    if [[ ${!name} != /* ]]; then printf 'cpro-env: %s must be absolute\n' "$name" >&2; errors=1; fi
  done
  [[ $errors == 0 ]] || return 1
  REPO_DIR=$(realpath -m -- "$REPO_DIR")
  AI_RUN_DIR=$(realpath -m -- "$AI_RUN_DIR")
  if [[ $AI_RUN_DIR == "$REPO_DIR" || $AI_RUN_DIR == "$REPO_DIR/"* || $AI_RUN_DIR == / ]]; then
    printf '%s\n' 'cpro-env: AI_RUN_DIR must be a dedicated directory outside the repository' >&2
    return 1
  fi
  if [[ $config == "$REPO_DIR/"* || $config == "$AI_RUN_DIR/"* || $config == "$AI_RUN_DIR" ]]; then
    printf '%s\n' 'cpro-env: keep the private config outside the repository and AI_RUN_DIR' >&2
    return 1
  fi
  files=("$REPO_DIR/targets/cpro/scripts/run_fused_probes.py"
         "$CANN_ROOT/include/graph/graph.h" "$CANN_ROOT/lib64/libgraph.so"
         "$OHOS_NATIVE_SDK/build/cmake/ohos.toolchain.cmake"
         "$HIAI_INCLUDE_DIR/HiAiModelManagerService.h")
  if [[ $HIAI_LIBRARIES == :* || $HIAI_LIBRARIES == *: || $HIAI_LIBRARIES == *::* ]]; then
    printf '%s\n' 'cpro-env: HIAI_LIBRARIES has an empty path entry' >&2; errors=1
  fi
  IFS=: read -r -a libraries <<< "$HIAI_LIBRARIES"
  files+=("${libraries[@]}")
  for path in "${files[@]}"; do
    if [[ $path != /* || ! -f $path ]]; then printf 'cpro-env: missing file: %s\n' "$path" >&2; errors=1; fi
  done
  for path in "$MODEL_PYTHON" "$CPRO_DDK_ROOT/tools/tools_omg/master/omg" "$OHOS_NATIVE_SDK/llvm/bin/clang++"; do
    if [[ ! -f $path || ! -x $path ]]; then printf 'cpro-env: missing executable: %s\n' "$path" >&2; errors=1; fi
  done
  if [[ ! -d $OHOS_NATIVE_SDK/sysroot ]]; then printf 'cpro-env: SDK must point to native/, missing sysroot: %s\n' "$OHOS_NATIVE_SDK" >&2; errors=1; fi
  if [[ $SN == -* || $SN == *[[:space:]]* || $BIP == -* || $BIP == *[[:space:]]* || ! $BIP =~ ^.+:[0-9]{1,5}$ ]]; then
    printf '%s\n' 'cpro-env: invalid SN or BIP; BIP must be the HDC server IP:port' >&2; errors=1
  else
    port=${BIP##*:}
    if (( 10#$port < 1 || 10#$port > 65535 )); then printf '%s\n' 'cpro-env: HDC port must be 1..65535' >&2; errors=1; fi
  fi
  if ! command -v "$HDC" >/dev/null 2>&1; then printf 'cpro-env: HDC not found: %s\n' "$HDC" >&2; errors=1; fi
  [[ $errors == 0 ]] || return 1
  mkdir -p -- "$AI_RUN_DIR" || return 1

  # Per-run paths and runtime identity overrides belong to the invocation;
  # leave any caller-provided values untouched.
  CPRO_TARGET=kirinx90
  GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
  QWEN35_RESOURCES="$REPO_DIR/specs/resources.json"
  QWEN35_ENV_CONFIG=$config
  PYTHONDONTWRITEBYTECODE=1
  PYTHONPATH="$REPO_DIR/model:$REPO_DIR"
  # Prefer the selected interpreter for OMG workers; repeated sourcing is stable.
  IFS=: read -r -a old_path <<< "${PATH:-}"
  PATH="$(dirname -- "$MODEL_PYTHON")"
  for path in "${old_path[@]}"; do
    [[ -z $path || :$PATH: == *":$path:"* ]] || PATH+=":$path"
  done
  for name in "${required[@]}" "${optional[@]}" CPRO_TARGET GE_MODEL_TOOL QWEN35_RESOURCES \
              QWEN35_ENV_CONFIG PYTHONDONTWRITEBYTECODE PYTHONPATH PATH; do
    if [[ -n ${!name-} ]]; then printf 'export %s=%q\n' "$name" "${!name}"; else printf 'unset %s\n' "$name"; fi
  done
  printf 'cpro-env: loaded %s\n  project: %s\n  runs: %s\n  HDC server: %s  device: %s\n' "$config" "$REPO_DIR" "$AI_RUN_DIR" "$BIP" "$SN" >&2
)

if _qwen35_exports=$(_qwen35_load_env "${1:-}"); then
  eval "$_qwen35_exports"
  unset _qwen35_exports
  unset -f _qwen35_load_env
else
  unset _qwen35_exports
  unset -f _qwen35_load_env
  return 1
fi
