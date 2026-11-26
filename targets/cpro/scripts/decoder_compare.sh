#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
decoder_compare.sh {golden|air|compile|execute|diagnose|gate-probe|gate-report|transfer|prepare|board|source|plan} [--mode {linear|full|4|40|layerwise} --phase {prefill|decode}]
                   [--gdn-conv {conv2d|taps|original}] [--physical-convstate] [--no-patch|--native-activations|--silu-only|--softplus-only|--gdn-silu-only|--gdn-silu-site {conv|gate}|--all-silu-no-gate-cast] [--run-root DIR] [--remote-run DIR] [--patched-source DIR] [--recovery] [--replace-inputs] [--replace-weights]
source      Generate or select the editable CPro source directory.
golden      Generate CPU and 310P reference data.
air         Export CPro AIR from the golden case.
compile     Compile the local AIR package with OMG and record OMC I/O.
transfer {golden|air|omc}  Receive a stage from its producer on this downstream host.
            --local-source DIR reads a mounted upstream case directory.
execute     Run compiled OMCs through HDC and compare CPU / 310P / CPro.
diagnose    Inspect a retained board failure or four-layer accuracy mismatch.
gate-probe  Compile and run four small OMCs from the existing four-layer T64 RMS gate on the OMC/board host.
gate-report Compare retained AIR edges, kernel build logs and metadata for the full model and the silu-cast-mul probe.
prepare     On the combined reference/export host: capture reference, compute CPU golden, export static AIR.
board       On the DDK host: fetch, compile, run on device and compare CPU / 310P / CPro.
plan        Print the layers, sequence length and number of OMCs without running the model.
--replace-inputs   Upload all board inputs again. Otherwise reuse matching cached files.
--replace-weights  Upload all external board weights again. Both flags can be combined.
--original-board-args  Run the existing OMC with the owner's V2 command flags. This is the default for --gdn-conv original.
--no-patch  Export four-layer T64 prefill directly from CPRO_SOURCE with the native OMG and V2 settings.
--native-activations  Use stock SiLU and Softplus with the aligned four-layer T64 native GDN export.
--silu-only  Use FP16 SiluPatch and stock Softplus with the same four-layer T64 export.
--softplus-only  Use stock SiLU and FP16 SoftplusPatch with the same four-layer T64 export.
--gdn-silu-only  Use FP16 SiluPatch at both GDN sites, stock SiLU in every MoE, and FP16 SoftplusPatch in GDN.
--gdn-silu-site {conv|gate}  Keep SiluPatch at one GDN site in L0–L2. MoE uses stock SiLU and GDN uses SoftplusPatch.
--all-silu-no-gate-cast  Enable all GDN and MoE SiluPatch sites and SoftplusPatch. Omit the explicit FP32 cast on each GDN RMS gate output.
             The activation controls accept --gdn-conv conv2d for explicit NCHW Conv2D with native ConvState inputs.
--physical-convstate  Use the last four physical QKV rows for four-layer T64 ConvState and omit the position control.
                      Defaults to Conv2D and works with any patched activation selection.
--recovery  Reuse verified completed OMC outputs in layerwise mode when resuming on the same board.
DECODER_ENV may point to a host-local shell configuration file.
HELP
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    gate-report)
        shift
        exec bash "$0" compile --gate-probe report --mode 4 --phase prefill "$@" ;;
    gate-probe)
        shift
        bash "$0" compile --gate-probe --mode 4 --phase prefill "$@"
        exec bash "$0" execute --gate-probe --mode 4 --phase prefill "$@" ;;
    golden|air|compile|execute|diagnose|transfer|source|prepare|board|plan) action=$1; shift ;;
    *) usage >&2; exit 2 ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo"
config_dir=${XDG_CONFIG_HOME:-$HOME/.config}/qwen35-cpro
role=$action
if [[ $action == diagnose ]]; then role=execute; fi
if [[ $action == source ]]; then role=air; fi
if [[ $action == transfer ]]; then
    case ${1:-} in golden) role=air ;; air) role=compile ;; omc) role=execute ;; esac
fi
env_file=${DECODER_ENV:-$config_dir/$role.env}
if [[ -z ${DECODER_ENV:-} && ! -f $env_file ]]; then env_file="$config_dir/decoder.env"; fi
if [[ -f $env_file || -n ${DECODER_ENV:-} ]]; then
    set -a
    source "$env_file"
    set +a
fi
export REPO_DIR="$repo"
export PYTHONDONTWRITEBYTECODE=1
if [[ $action != plan && $action != source ]]; then
    export QWEN35_TRACE_COMMANDS=${QWEN35_TRACE_COMMANDS:-1}
fi
tool="$repo/targets/cpro/tools/decoder_cases.py"

exec python3 "$tool" "$action" "$@"
