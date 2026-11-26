"""Compare host-recorded V2 input slots across the two native four-layer runs."""
import json


def slot_rows(ports, roles, identities):
    return [dict(port=p['name'], role=roles[p['name']], shape=p['shape'], dtype=p['dtype'], data=digest)
            for p, digest in zip(ports, identities, strict=True)]


def remember_original_inputs(root):
    layer = root/'board/L00-L03'
    paths = [root/'board/package/L00-L03/case.json', layer/'omc/build.json', layer/'replay/result.json']
    if not all(p.is_file() for p in paths):
        return
    case, build, receipt = (json.loads(p.read_bytes()) for p in paths)
    if not (case.get('no_patch') and receipt.get('status') == 'EXECUTED' and receipt.get('exitcode') == 0
            and receipt.get('omc_sha256') == build['model']['sha256']
            and receipt.get('fixture_sha256') == build['stamp']['bundle']):
        return
    roles = case['inputs']
    ports = build['inputs']
    slot_map = layer/'replay/slots.json'
    if slot_map.is_file():
        ports = [ports[s['compiled_index']] for s in json.loads(slot_map.read_bytes())['inputs']]
    if any(p['name'] not in roles for p in ports):
        return
    rows = slot_rows(ports, roles, receipt['input_hashes'])
    if any(row['data'] != case['input_files'][row['role']]['sha256'] for row in rows):
        return
    saved = root/'original-board-inputs.json'
    tmp = saved.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(attempt=receipt.get('attempt'), inputs=rows), indent=2) + '\n')
    tmp.replace(saved)


def compare_original_inputs(root, ports, roles, identities):
    saved = root/'original-board-inputs.json'
    if not saved.is_file():
        print('[四层输入] 本机没有保留原版成功运行的输入记录，跳过跨运行对照。', flush=True)
        return
    previous = json.loads(saved.read_bytes())
    rows = slot_rows(ports, roles, identities)

    def slots(records):
        return [(r['role'], r['dtype'], tuple(r['shape']), r['data']) for r in records]

    old, new = slots(previous['inputs']), slots(rows)
    print(f'[四层输入] 对照原版 V2 记录 {previous["attempt"]}，原版 {len(old)} 个槽位，本次 {len(new)} 个。')
    if old == new:
        print('所有槽位的输入角色、shape、dtype 和已记录的 bin 内容标识一致。', flush=True)
        return
    if sorted(old) == sorted(new):
        print('输入张量相同，编译后的槽位顺序有调整。')
    for i in range(max(len(old), len(new))):
        a = old[i] if i < len(old) else None
        b = new[i] if i < len(new) else None
        if a != b:
            print(f'  slot {i}: 原版 {a and a[:3]}，本次 {b and b[:3]}，'
                  f'bin 内容标识相同={bool(a and b and a[3] == b[3])}', flush=True)
