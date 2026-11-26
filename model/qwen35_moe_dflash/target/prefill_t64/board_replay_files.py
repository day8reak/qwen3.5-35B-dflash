"""Stage the current OMC and reuse manifest-identified files on a locked board."""
import re
import shlex

from ...execution.hdc_run import _relative
from ...execution.lock import _space


def shell_steps(hdc, directory, commands, label):
    # HDC shell command length varies by client. A full decoder has 89 inputs.
    prefix = f'set -e\ncd {shlex.quote(str(directory))}\n'
    pending = ''
    for command in commands:
        if pending and len(pending) + len(command) > 2400:
            hdc.shell(prefix + pending, label)
            pending = ''
        pending += command + '\n'
    if pending:
        hdc.shell(prefix + pending, label)


def deploy(hdc, root, token, tool, model, assets, output_bytes, replace):
    blobs = {}
    for path, key in assets:
        if not re.fullmatch(r'(?:input|weight)-[0-9]+-[a-f0-9]{64}', key):
            raise ValueError(f'{path}: missing artifact identity')
        size = path.stat().st_size
        blobs[key] = (path, size)

    # The global device lock is held. Only replace work inside a directory
    # previously created by this workflow. Keep a failed execution until this point.
    _, inventory = hdc.shell(
        f'set -e\ntest -x {tool}\n'
        f'if mkdir {root} 2>/dev/null; then echo v1 > {root}/.qwen35-replay-cache; fi\n'
        f'test -d {root} && test ! -L {root} || exit 1\ncd {root}\n'
        'test -f .qwen35-replay-cache && test ! -L .qwen35-replay-cache && '
        'test "$(cat .qwen35-replay-cache)" = v1 || exit 1\n'
        'test ! -L cache\nmkdir -p cache\n'
        'rm -rf work incoming\nmkdir work work/outputs incoming\n'
        f'echo {token} > work/.qwen35-hdc-owner\n'
        'cd cache\nfor f in *; do if test "$f" = "*" && test ! -e "$f"; then continue; fi\n'
        'test -f "$f" && test ! -L "$f" || exit 1\nstat -c "%n %s" "$f" || exit 1\ndone',
        'prepare-board-cache')
    cache = {}
    for line in inventory.splitlines():
        hit = re.fullmatch(r'((?:input|weight)-[0-9]+-[0-9a-f]{64}) ([0-9]+)', line)
        if not hit:
            raise ValueError(f'Unexpected board cache entry in {root}: {line!r}')
        cache[hit[1]] = int(hit[2])

    hits = {key for key, (_, size) in blobs.items()
            if cache.get(key) == size and key.partition('-')[0] not in replace}
    shell_steps(hdc, root, [f'rm cache/{key}' for key in cache if key not in hits], 'prune-board-cache')
    missing = {key: item for key, item in blobs.items() if key not in hits}
    model_size = model.stat().st_size
    upload_bytes = model_size + sum(size for _, size in missing.values())
    _, space = hdc.shell(f'df -Pk {root}', 'preflight')
    if _space(space) < upload_bytes + output_bytes:
        raise OSError(f'{root}: insufficient board space for {upload_bytes:,} upload and {output_bytes:,} output bytes')
    print(f'[{root.name}] 板端复用 {len(blobs)-len(missing)}/{len(blobs)} 个权重或输入文件，'
          f'上传 OMC 并补传 {len(missing)} 个文件，共 {upload_bytes:,} 字节', flush=True)

    upload = [(model, 'model.omc'), *((path, key) for key, (path, _) in missing.items())]
    hdc.send_tree(upload, root/'incoming')
    # An interrupted directory transfer can leave a correctly named partial file.
    # Publish a blob only after HDC returns and the expected byte count is present.
    moves = []
    for key, (_, size) in missing.items():
        moves.append(f'test "$(stat -c %s incoming/{key})" = {size} && mv incoming/{key} cache/{key} || exit 1')
    moves.extend([f'test "$(stat -c %s incoming/model.omc)" = {model_size}',
                  'mv incoming/model.omc work/model.omc', 'rmdir incoming'])
    shell_steps(hdc, root, moves, 'install-board-cache')
    return dict(reused_files=len(blobs)-len(missing), uploaded_assets=list(missing),
                uploaded_bytes=upload_bytes, forced=sorted(replace))


def bind_files(hdc, work, files):
    # Links keep each weight at the exact relative path encoded in the OMC.
    commands = []
    parents = set()
    for key, relative in files:
        rel = _relative(str(relative))
        parent = rel.parent
        if parent.parts and parent not in parents:
            commands.append('mkdir -p ' + shlex.quote(str(parent)))
            parents.add(parent)
        commands.append(f'ln ../cache/{key} {shlex.quote("./" + str(rel))}')
    shell_steps(hdc, work, commands, 'bind-board-files')
