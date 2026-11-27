#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
decoder_compare.sh {golden|air|compile|execute|diagnose|l0-embedding|l0-steps-reference|l0-steps|l0-steps-same-input|l0-input-rmsnorm-air|l0-input-rmsnorm-omc|l0-input-rmsnorm|l0-shared-gate-reference|l0-shared-gate|l0-shared-gate-fp16|gdn-export|gdn-whole|gdn-bisect|gate-probe|gate-upstream|gate-report|transfer|prepare|board|source|plan} [--mode {linear|full|4|40|layerwise} --phase {prefill|decode}]
                   [--gdn-conv {conv2d|taps|original}] [--physical-convstate] [--no-patch|--native-activations|--silu-only|--softplus-only|--gdn-silu-only|--gdn-silu-site {conv|gate}|--all-silu-no-gate-cast|--rms-silu-layers LAYERS|--rms-silu-token-view] [--run-root DIR] [--remote-run DIR] [--patched-source DIR] [--recovery] [--replace-inputs] [--replace-weights]
source      Generate or select the editable CPro source directory.
golden      Generate CPU and 310P reference data.
air         Export CPro AIR from the golden case.
compile     Compile the local AIR package with OMG and record OMC I/O.
transfer {golden|air|omc}  Receive a stage from its producer on this downstream host.
            --local-source DIR reads a mounted upstream case directory.
execute     Run compiled OMCs through HDC and compare CPU / 310P / CPro.
diagnose    Inspect a retained board failure or four-layer accuracy mismatch.
gate-probe  Compile and run four small OMCs from the existing four-layer T64 RMS gate on the OMC/board host.
gate-upstream  Reconnect the first gate to its original z projection, RMSNorm, then both. Compile and run three OMCs.
gdn-export  Export one L0 linear attention AIR from the retained four-layer T64 physical-history package.
gdn-whole   Export the same L0 module, compile one OMC and replay on the board with synthetic inputs.
gdn-bisect  Cut the retained standalone L0 GDN at multiple depths, compile OMCs, replay them and report minimal reproductions.
l0-embedding  Observe the 310P embedding path up to L0 and compare its token rows with the retained CPro inputs.
l0-steps-reference  Capture CPU and 310P L0 intermediates for the existing four-layer YAML prompt.
l0-steps    Read out L0 from the existing four-layer AIR, run with its real inputs and write three-way matrices.
l0-steps-same-input   Run the L0 readouts with the saved 310P hidden as the CPro decoder input.
l0-input-rmsnorm-air   Extract the single L0 input RMSNorm AIR, its operands and the captured 310P output.
l0-input-rmsnorm-omc   Compile single-op/export.air to single-op/export.omc. Use --rmsnorm-air custom.air for a custom op.
l0-input-rmsnorm       Export, compile and run this RMSNorm on the same 310P input.
l0-shared-gate-reference   Export the L0 scalar gate weight and saved reference rows on the golden host.
l0-shared-gate   Compare the saved gate logits and weighted outputs offline on the CPro host.
l0-shared-gate-fp16   Run one same-input L0 shared-router OMC with FP16 scalar MatMul and an FP32 cast before sigmoid.
gate-report Compare AIR and OMC gate edges, kernel build logs and metadata for the full model and the silu-cast-mul probe.
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
--rms-silu-token-view  Keep every custom activation. Run the three four-layer RMS gate SiluPatch kernels on FP16 [64,4096].
--rms-silu-layers LAYERS  Use SiluPatch on these RMS gates (none, 0, 1, 2, 0,1, 0,2, 1,2, or 0,1,2).
             Other RMS gates use FP16 F.silu. Convolution and MoE keep SiluPatch, decay keeps SoftplusPatch.
             The activation controls accept --gdn-conv conv2d for explicit NCHW Conv2D with native ConvState inputs.
--physical-convstate  Use the last four physical QKV rows for four-layer T64 ConvState and omit the position control.
                      Defaults to Conv2D and works with any patched activation selection.
--recovery  Reuse verified completed OMC outputs in layerwise mode when resuming on the same board.
DECODER_ENV may point to a host-local shell configuration file.
HELP
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    l0-embedding)
        shift
        exec bash "$0" golden --gate-probe l0-embedding --mode 4 --phase prefill "$@" ;;
    l0-shared-gate-reference)
        shift
        exec bash "$0" golden --gate-probe l0-shared-gate --mode 4 --phase prefill "$@" ;;
    l0-shared-gate)
        shift
        exec bash "$0" compile --gate-probe l0-shared-gate --mode 4 --phase prefill "$@" ;;
    l0-steps-reference)
        shift
        exec bash "$0" golden --gate-probe l0-steps --mode 4 --phase prefill "$@" ;;
    l0-input-rmsnorm-air|l0-input-rmsnorm-omc)
        mode=$1
        shift
        exec bash "$0" compile --gate-probe "$mode" --mode 4 --phase prefill "$@" ;;
    l0-steps|l0-steps-same-input|l0-shared-gate-fp16|l0-input-rmsnorm)
        mode=$1
        shift
        bash "$0" compile --gate-probe "$mode" --mode 4 --phase prefill "$@"
        exec bash "$0" execute --gate-probe "$mode" --mode 4 --phase prefill "$@" ;;
    gate-report)
        shift
        exec bash "$0" compile --gate-probe report --mode 4 --phase prefill "$@" ;;
    gdn-export)
        shift
        exec bash "$0" compile --gate-probe gdn-export --mode 4 --phase prefill "$@" ;;
    gate-probe|gate-upstream|gdn-whole|gdn-bisect)
        gate_mode=slices
        if [[ $1 == gate-upstream ]]; then gate_mode=upstream; fi
        if [[ $1 == gdn-* ]]; then gate_mode=$1; fi
        shift
        bash "$0" compile --gate-probe "$gate_mode" --mode 4 --phase prefill "$@"
        exec bash "$0" execute --gate-probe "$gate_mode" --mode 4 --phase prefill "$@" ;;
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
