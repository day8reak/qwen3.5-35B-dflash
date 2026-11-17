"""Connect the shared packed C16 Draft to the resident loop, without compilation.

Admission checks metadata and file extents, not numerical accuracy. Payload
digests are existing deployment labels under the size-and-ABI runtime policy.
"""
from pathlib import Path

from qwen35_moe_dflash.execution.runtime_config import read_json, file_record, write_json
from .runtime import GEOMETRY, validate_abi

VARIANT = 'packed-context-kv-v1'
KIND = 'qwen35-incremental-draft-pytorch-omc'


def _child(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('packed Draft file must be relative to its bundle')
    return root / relative


def inspect_bundle(directory, prepared, *, ddk_id):
    """Read the published baseline and bind its original host embedding table."""
    from .air_feature_pack import check_contract
    from .host_embedding import source_contract
    root, prepared = Path(directory).resolve(), Path(prepared).resolve()
    source = read_json(root / 'manifest.json')
    publication = source.get('publication', {})
    if (source.get('artifact_kind') != KIND or source.get('status') != 'PYTORCH_DRAFT_OMC_COMPILED'
            or source.get('ddk_id') != ddk_id or source.get('weight_storage') != 'embedded-stream'
            or source.get('compiled_abi') != 'BOUND_DIAGNOSTIC'
            or publication.get('policy') != 'pytorch-draft-shared-v1'
            or source.get('optimization', {}).get('policy') != 'packed-spatial-host-embedding'
            or source.get('top1', {}).get('policy') != 'stable-arg-max'
            or source.get('precision_repair', {}).get('policy') != 'source-fp32-rope-trig-before-gather-v1'
            or source.get('precision_repair', {}).get('status') != 'APPLIED' or source.get('offline_weights')):
        raise ValueError('expected the shared packed/StableArgMax/RoPE baseline from the selected DDK')
    geometry = validate_abi(source)
    if geometry != dict(context_rows=16, cache_capacity=192, max_draft_tokens=15) or not check_contract(source):
        raise ValueError('resident packed baseline requires C16/cap192/K15 and CPU feature pack')
    proof = read_json(_child(root, publication['abi']))
    plan = read_json(prepared / 'prepared.json')
    embedding = source_contract(plan)
    if (proof.get('artifact_kind') != 'qwen35-pytorch-draft-diagnostic-abi'
            or proof.get('status') != 'PASSED'
            or any(proof.get(k) != source[k] for k in ('inputs', 'outputs'))
            or proof.get('packed_model_sha256') != source['model']['sha256']
            or plan.get('draft_config') != source.get('draft_config')
            or embedding != source.get('host_embedding') or embedding != proof.get('host_embedding')
            or embedding['weight']['shape'] != [248320, 2048]):
        raise ValueError('packed Draft ABI, model, or prepared FP16 embedding contract differs')
    model = file_record(_child(root, source['model']['path']), size=source['model']['bytes'],
                        cache_key=source['model']['sha256'])
    row = plan['weights']['target.embed_tokens.weight']
    table = file_record(_child(prepared, row['path']), size=row['bytes'], cache_key=row['sha256'])
    return source, model, table


def bind(directory, prepared, output, *, ddk_id, max_draft_tokens, max_proposals, total_tokens):
    source, model, table = inspect_bundle(directory, prepared, ddk_id=ddk_id)
    geometry = {key: source[key] for key in GEOMETRY}
    if (type(max_draft_tokens) is not int or max_draft_tokens != geometry['max_draft_tokens']
            or type(max_proposals) is not int or not 1 <= max_proposals <= max_draft_tokens
            or total_tokens - 1 > geometry['cache_capacity']):
        raise ValueError('request candidate gear or context capacity differs from packed Draft')
    native = dict(schema_version=1, variant=VARIANT, model_sha256=model['sha256'],
        inputs=source['inputs'], outputs=source['outputs'], **geometry,
        preparation=dict(host_embedding='prepared-fp16-block-row-lookup-v1',
            feature_pack='cpu-feature-pack-c16-v1', mask_token_id=source['host_embedding']['mask_token_id'],
            embedding_sha256=table['sha256']))
    path = Path(output) / 'incremental-draft-abi.json'
    write_json(path, native)
    return dict(model_source=model, embedding_table=table, abi=file_record(path),
        source_bundle=str(Path(directory).resolve()), prepared_source=str(Path(prepared).resolve()),
        variant=VARIANT, inputs=source['inputs'], outputs=source['outputs'], ddk_id=ddk_id,
        optimization=source['optimization'], feature_pack=source['feature_pack'],
        host_embedding=source['host_embedding'], **geometry,
        numerical_validation='NOT_VALIDATED', device_integration='NOT_RUN')
