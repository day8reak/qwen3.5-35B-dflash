"""Actual native BPE/text frontend vs the locked HF tokenizer; no HIAI device.

Set QWEN35_TOKENIZER_SOURCE_DIR and QWEN35_TOKENIZER_TEST_BINARY to run the
asset-dependent checks. Assets are checked against specs/tokenizer.lock.json.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import pytest

from qwen35_moe_dflash import board_tokenizer as tok


@pytest.fixture(scope='module')
def assets(tmp_path_factory):
    source = os.environ.get('QWEN35_TOKENIZER_SOURCE_DIR')
    binary = os.environ.get('QWEN35_TOKENIZER_TEST_BINARY')
    if not source or not binary:
        pytest.skip('explicit pinned tokenizer assets/native utility required')
    root = tmp_path_factory.mktemp('tokenizer')
    bundle = root / 'tokenizer.bin'; tok.build_bundle(source, bundle)
    return Path(source), Path(binary), bundle, tok.reference(source)


def call(assets, mode, lines):
    _, binary, bundle, _ = assets
    run = subprocess.run([str(binary), str(bundle), mode], input=''.join(line+'\n' for line in lines),
                         text=True, capture_output=True, timeout=90)
    assert run.returncode == 0, run.stderr
    result = run.stdout.splitlines()
    assert len(result) == len(lines)
    return result


@pytest.mark.parametrize('mode', ['raw', 'chat'])
def test_original_encoding_unicode_and_random_text(assets, mode):
    rng = random.Random(42)
    corpus = ['你好，请介绍一下自己。', "Hello, I'm happy! 123.45\n\n", 'e\u0301 café 👨\u200d👩\u200d👧\u200d👦 🇨🇳',
        '日本語、한국어 العربية हिन्दी ไทย', ' \t\r\n', "'ſ 'S 'LL", '<think>hello</think>',
        '\u001c\u00a0trim\u3000', '<|im_start|>user\n你好<|im_end|>']
    data = json.loads((assets[0] / 'tokenizer.json').read_text())
    corpus += [r['content'] for r in data['added_tokens']]
    corpus += [''.join(rng.choice('你好中文hello WORLD 12345😄!?\n\té\u0301\u0323')
                       for _ in range(rng.randrange(1,200))) for _ in range(1000)]
    # Random scalar values cover UTF-8 widths and non-ASCII regex classes.
    scalar = lambda: rng.choice([rng.randrange(1,0xd800),rng.randrange(0xe000,0x110000)])
    corpus += [''.join(chr(scalar()) for _ in range(20)) for _ in range(300)]
    rows = call(assets, mode, [s.encode().hex() for s in corpus])
    for text, row in zip(corpus, rows):
        expected = assets[3].encode(tok.render(text, mode), add_special_tokens=False).ids
        assert ([int(x) for x in row.split(',')] if row else []) == expected, repr(text)


@pytest.mark.parametrize('skip', [True, False])
def test_original_decode_partial_utf8_and_special_ids(assets, skip):
    rng = random.Random(7)
    ids = [[rng.randrange(248070) for _ in range(rng.randrange(1,25))] for _ in range(3000)]
    ids += [[i] for i in range(248044,248070)]
    ids += [assets[3].encode('你好👩🏽‍🚀').ids[:i] for i in range(1,8)]
    rows = call(assets, 'decode' if skip else 'decode-all', [','.join(map(str,x)) for x in ids])
    for tokens, row in zip(ids, rows):
        assert bytes.fromhex(row).decode() == assets[3].decode(tokens, skip_special_tokens=skip), tokens


def test_padded_head_id_is_explicit_replacement(assets):
    row, = call(assets, 'decode', ['248319'])
    assert bytes.fromhex(row).decode() == '\ufffd'


def test_single_user_chat_matches_locked_jinja(assets):
    from jinja2 import Environment
    template = Environment().from_string((assets[0] / 'chat_template.jinja').read_text())
    for text in ('你好', '  a\n\u3000', 'é🙂', ''):
        expected = template.render(messages=[dict(role='user',content=text)], add_generation_prompt=True, enable_thinking=False)
        if text: assert tok.render(text) == expected


@pytest.mark.parametrize('text', ['', '\0bad', 'x'*16385])
def test_invalid_text_rejected(text):
    with pytest.raises(ValueError): tok.render(text)


def test_single_tool_response_is_not_a_chat_query():
    with pytest.raises(ValueError, match='only user query'):tok.render('<tool_response>test</tool_response>')


def test_text_budget_rejected_before_model_preflight(assets, tmp_path, monkeypatch):
    from qwen35_moe_dflash import board_serial_dflash as job
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    monkeypatch.setattr(job, '_backends', lambda: pytest.fail('oversized text must fail before model inspection'))
    with pytest.raises(ValueError, match='exceeds the current S64 limit'):
        job.prepare_board_serial(decode_job='unused',runtime_probe='unused',state_lineage='unused',
            source_torchair_pb2='unused',draft_bundle='unused',weight_table='unused',scale_table='unused',
            output=tmp_path/'oversize',text='你好',tokenizer_dir=assets[0],resident=True,prefill_feedback='unused',max_new_tokens=60)


def test_bundle_tamper_and_malformed_utf8(assets, tmp_path):
    binary, bundle = assets[1:3]
    raw = bytearray(bundle.read_bytes()); raw[:8] = b'BADTOK00'
    bad = tmp_path / 'bad.bin'; bad.write_bytes(raw)
    result = subprocess.run([str(binary),str(bad),'raw'],input='61\n',text=True,capture_output=True)
    assert result.returncode != 0 and 'header' in result.stderr
    result = subprocess.run([str(binary),str(bundle),'raw'],input='eda080\n',text=True,capture_output=True)
    assert result.returncode != 0 and 'UTF-8' in result.stderr


def test_offline_prepare_cli_never_downloads(assets, tmp_path, monkeypatch, capsys):
    import runpy
    import socket
    import sys
    def network_forbidden(*args, **kwargs):
        pytest.fail('offline tokenizer preparation attempted network access')
    monkeypatch.setattr(tok, 'acquire', network_forbidden)
    monkeypatch.setattr(tok.urllib.request, 'urlopen', network_forbidden)
    monkeypatch.setattr(socket.socket, 'connect', network_forbidden)
    script=Path(__file__).parents[1]/'targets/cpro/scripts/prepare_board_tokenizer.py'
    output=tmp_path/'offline-prepared'
    monkeypatch.setattr(sys, 'argv', [str(script),'--source-dir',str(assets[0]),'--output',str(output)])
    runpy.run_path(str(script),run_name='__main__')
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='BOARD_TOKENIZER_PREPARED'
    assert result['source_dir']==str(assets[0].resolve())
    assert (output/'tokenizer.bin').read_bytes()==assets[2].read_bytes()
    assert result['source']==json.loads(tok.LOCK.read_text())


@pytest.mark.parametrize('mode', ['chat', 'raw'])
@pytest.mark.parametrize('preread', [False, True])
def test_text_enters_actual_native_prefill_loop_and_recovers(assets, tmp_path, monkeypatch, mode, preread):
    from test_board_serial_integration import make_case, enable_resident_mixed, enable_compact_memory, execute, _recover_actual_mixed_case
    from qwen35_moe_dflash.board_serial_dflash import make_config
    binary = os.environ.get('QWEN35_RESIDENT_MIXED_TEST_BINARY')
    if not binary: pytest.skip('explicit actual mixed native/fake SDK binary required')
    assert hashlib.sha256(Path(binary).read_bytes()).hexdigest() == os.environ['QWEN35_RESIDENT_MIXED_TEST_BINARY_SHA256']
    case = make_case(tmp_path/'text-loop',(Path(binary),Path(binary)),dump=False)
    enable_resident_mixed(case); enable_compact_memory(case)
    ids, _, info = tok.prepare_text(assets[0], '你好，请介绍自己。', mode, case['root'])
    manifest = case['manifest']; manifest['text_io']=info;manifest['prompt_token_ids']=ids
    if preread: manifest['draft_preread']='sequential-1m-v1'
    manifest['timeout_seconds'] = 60
    from qwen35_moe_dflash.board_prefill_binding import control_assumption
    manifest['prefill']['experimental_control_assumption']=control_assumption(len(ids))
    paths = {key:str(path) for key,path in case['files'].items()}
    paths.update(target_library_path=str(case['root']),draft_library_path=str(case['root']),
        resident_abi=manifest['resident_abi']['path'], prefill_abi=manifest['prefill_abi']['path'],
        prefill_model=manifest['prefill']['model']['path'],tokenizer_bundle=info['bundle']['path'],prompt_text=info['prompt']['path'])
    case['config'].write_bytes(make_config(manifest, paths))
    monkeypatch.setenv('FAKE_HIAI_TEXT_PROMPT_IDS', ','.join(map(str,ids)))
    result = execute(case)
    assert result.returncode == 0, result.stderr
    loop=json.loads((case['output']/'loop.json').read_text())
    assert loop['prompt']==ids and loop['counts']['prefill_native_calls']==1
    assert loop['counts']['target_native_calls']==8 and loop['counts']['draft_native_calls']>=2
    assert loop['text_io']['execution']=='BOARD_LOCAL_ENCODE_AND_DECODE'
    assert loop['text_io']['host_admission_ids_matched'] is True
    assert loop['text_latency']['load_and_encode_us']>0
    if preread:
        load=loop['resident_lifecycle']['draft_v1_lifecycle']['load_profile']
        assert load['preread_bytes']==manifest['draft']['model_source']['bytes']
        assert [row['phase'] for row in load['phases']].count('MODEL_PREREAD')==1
        assert all(row['completed'] for row in load['phases'])
        assert sum(row['wall_us'] for row in load['phases']) <= (loop['resident_lifecycle']['draft_load_ms']+1)*1000
        assert loop['resident_lifecycle']['draft_loads']==loop['resident_lifecycle']['target_loads']==1
    tok.validate_text_result(manifest, loop)
    _recover_actual_mixed_case(case, loop, 0, monkeypatch)
    recovered=json.loads((case['root']/'recovered/result.json').read_text())
    assert recovered['generated_text']==loop['generated_text']
    assert recovered.get('draft_preread')==manifest.get('draft_preread')
    assert set(p.name for p in case['output'].iterdir())=={'loop.json','progress.json','retrieval-manifest.tsv'}
    loop['generated_text']='corrupted'
    with pytest.raises(ValueError,match='decode differs'):tok.validate_text_result(manifest,loop)


def test_board_admission_mismatch_never_loads_models(assets,tmp_path):
    from test_board_serial_integration import make_case, enable_resident_mixed, enable_compact_memory, execute
    binary=os.environ.get('QWEN35_RESIDENT_MIXED_TEST_BINARY')
    if not binary:pytest.skip('explicit mixed native binary required')
    case=make_case(tmp_path/'bad-admission',(Path(binary),Path(binary)),dump=False)
    enable_resident_mixed(case);enable_compact_memory(case)
    _,_,info=tok.prepare_text(assets[0],'你好','chat',case['root'])
    rows=case['config'].read_text()+'text_io\tqwen35-text-v1\nprompt_format\tchat\n'
    for role,key in [('tokenizer_bundle','bundle'),('prompt_text','prompt')]:
        r=info[key];rows+='\t'.join(map(str,(role,r['path'],r['sha256'],r['bytes'])))+'\n'
    case['config'].write_text(rows);trace=case['root']/'sdk.trace'
    result=execute(case,native_trace=trace)
    assert result.returncode!=0 and 'admission IDs' in result.stderr
    assert not trace.exists()
