"""Bounded reproduction of the inspected Linux OMG wrapper's default branch.

The caller supplies an already sourced shell. Vendor scripts are read and hash
checked, never executed; their chmod/mkdir/loader-link side effects are omitted.
This profile diagnoses entry differences and does not establish an X90 target.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

from qwen35_moe_dflash.common.hashing import sha256_file


DEFAULT_PROFILE = 'isolated-kirinx90'
SOURCED_PROFILE = 'sourced-wrapper-default'
COMPILER_PROFILES = (DEFAULT_PROFILE, SOURCED_PROFILE)
UNSPECIFIED_PLATFORM = 'UNSPECIFIED_COMPILER_DEFAULT'
# Scripts inspected in the user's selected DDK, not an assumption about other releases.
SCRIPT_HASHES = {
    'tools/tools_ascendc/set_ascendc_env.sh': '6d99a9a842cfeece87a57bd782093908be0bc3c783543fb13173ba6d54fd16df',
    'tools/tools_omg/omg': 'a0deee79968b5deed79ca5ae5839014c1d7ecc55ea530ef84996b64712ed4bb4',
}


def validate_profile(profile, tbe_diagnosis=None):
    if profile not in COMPILER_PROFILES:
        raise ValueError(f'compiler_profile must be one of {COMPILER_PROFILES}')
    if profile == SOURCED_PROFILE and tbe_diagnosis is not None:
        raise ValueError('sourced-wrapper-default requires the sourced environment without --tbe-diagnosis')


def sourced_wrapper_environment(ddk_root, inherited=None):
    """Model only master + Linux + no platform/execute-device/version override."""
    if sys.platform != 'linux':
        raise ValueError('sourced-wrapper-default is bound to the inspected Linux entry')
    ddk = Path(ddk_root).resolve()
    env = dict(os.environ if inherited is None else inherited)
    if not env.get('DDK_PATH') or Path(env['DDK_PATH']).resolve() != ddk:
        raise ValueError('sourced-wrapper-default requires DDK_PATH from the selected DDK sourced shell')
    scripts = {}
    for relative, expected in SCRIPT_HASHES.items():
        path = (ddk / relative).resolve()
        if ddk not in path.parents or not path.is_file() or path.stat().st_size > 256 << 10:
            raise ValueError(f'inspected compiler entry is missing or outside DDK: {relative}')
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f'compiler entry changed; inspect before using sourced-wrapper-default: {relative}')
        scripts[relative] = dict(sha256=digest, bytes=path.stat().st_size)
    master = ddk / 'tools/tools_omg/master'
    omg = master / 'omg'
    if ddk not in omg.resolve().parents or not omg.is_file():
        raise ValueError('sourced-wrapper-default requires the selected master/omg')
    with omg.open('rb') as stream:
        if stream.read(4) != b'\x7fELF':
            raise ValueError('sourced-wrapper-default requires master/omg to be an ELF executable')
    env['LD_LIBRARY_PATH'] = ':'.join(str(p) + '/' for p in
        (master / 'lib64', ddk / 'ddk/tbe/lib64', ddk / 'ddk/ascendcpp/lib64'))
    env['CCE_LIB_DIR'] = str(master / 'lib64') + '/'
    env['KIRIN_CONF_JSON_PATH'] = str(ddk / 'ddk/ascendcpp/conf') + '/'
    env['PATH'] = str(ddk / 'ddk/ccec_compiler/bin') + '/:' + env.get('PATH', '')
    # Shell assignment preserves export only when the variable was inherited.
    if 'HIAI_VERSION' in env:
        env['HIAI_VERSION'] = 'master'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    evidence = dict(
        scripts=scripts, script_execution='NOT_RUN',
        branch='LINUX_MASTER_NO_PLATFORM_OR_VERSION_OR_EXECUTE_DEVICE_OVERRIDE',
        scope='ENTRY_ENVIRONMENT_AND_OPTIONS_COMPARISON; NOT_HISTORICAL_ENV_RECONSTRUCTION',
        inherited_environment='CURRENT_CALLER; DDK_PATH_MATCHED; SOURCE_EXECUTION_NOT_ATTESTED',
        preserved_before_ipc_setup=['PYTHONPATH', 'SOC_VERSION'],
        deviations=['VENDOR_CHMOD_MKDIR_AND_LOADER_LINK_WRITES_OMITTED',
                    'PYTHONDONTWRITEBYTECODE=1',
                    'EXISTING_RUN_LOCAL_IPC_TEMP_AND_MULTIPROCESSING_OVERLAY; SEE_OMG_ENVIRONMENT_JSON',
                    'ARGV_PASSED_DIRECTLY_WITHOUT_WRAPPER_SPACE_REMOVAL_OR_SHELL_GLOBBING'],
        target_verification='UNVERIFIED; NO_PLATFORM_OR_TARGET_ARGUMENT')
    return omg, env, evidence
