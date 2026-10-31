#!/usr/bin/env python3
"""Fetch pinned original tokenizer metadata and build the portable board bundle."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from qwen35_moe_dflash.board_tokenizer import acquire, build_bundle

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--source-dir', help='Already downloaded exact original tokenizer assets; otherwise fetch three pinned public metadata files')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    source = Path(a.source_dir) if a.source_dir else out / 'source'
    if not a.source_dir: acquire(source)
    result = build_bundle(source, out / 'tokenizer.bin')
    (out / 'manifest.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='BOARD_TOKENIZER_PREPARED', source_dir=str(source.resolve()), **result)))
