"""Read-only compilation evidence checks with synthetic graph schemas."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qwen35_moe_dflash.target.layer_probe import inspection
from qwen35_moe_dflash.target.layer_probe.common import KIND, identity, save_json, sha


class FakeSchema:
    package = 'ge.proto'
    enums = {'ge.proto.DataType': {'DT_FLOAT16': 1, 'DT_INT32': 3}}
    evidence = {'scope': 'host fake schema'}

    def __init__(self, *args, **kwargs):
        pass

    def decode(self, raw, kind):
        if kind != 'ModelDef' or raw not in (b'SOURCE', b'IR'):
            raise ValueError('not a synthetic ModelDef')
        shape = {} if raw == b'SOURCE' else {'dim': [256, 1024]}
        graph = dict(name='region', op=[
            dict(name='weight', type='Const', attr=[dict(key='value', value={'t': {'data': 'PRIVATE_PAYLOAD'}})],
                 output_desc=[dict(shape={'dim': [2048, 1024]}, dtype=3)]),
            dict(name='qmatmul', type='QuantBatchMatmulV4444', input=['x:0', 'weight:0'],
                 attr=[dict(key='transpose_x2', value={'b': False})],
                 output_desc=[dict(shape=shape, dtype=1)]),
            dict(name='reshape', type='Reshape', input=['qmatmul:0'],
                 output_desc=[dict(shape={'dim': [1, 256, 8192]}, dtype=1)]),
            dict(name='out', type='NetOutput', input=['reshape:0'])])
        return dict(graph=[graph])


class InspectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AI_RUN_DIR'])
        self.root = Path(self.temp.name)
        self.source, self.compiled = self.root/'air', self.root/'omc'
        self.name = 'l00_c00_linear_attn_in_proj_qkv_o0'
        self.folder = self.source/'models'/self.name
        self.folder.mkdir(parents=True)
        (self.folder/'export.air').write_bytes(b'SOURCE')
        (self.folder/'air-schema.pb').write_bytes(b'host descriptor')
        (self.folder/'weight').mkdir()
        self.weight = self.folder/'weight/large-weight'
        self.weight.write_bytes(b'NEVER_READ_WEIGHT')
        save_json(self.folder/'export.json', dict(status='EXPORTED',
            schema={'schema_file': 'ge_ir.proto', 'schema_package': 'ge.proto'},
            air=identity(self.folder/'export.air', self.folder),
            source_inputs=[dict(name='x', shape=[1, 256, 2048], dtype='float16')],
            golden=dict(shape=[1, 256, 8192], dtype='float16')))
        manifest = dict(artifact_kind=KIND, status='EXPORTED', cases=[dict(id=self.name,
            status='EXPORTED', export=identity(self.folder/'export.json', self.source))])
        save_json(self.source/'manifest.json', manifest)
        save_json(self.compiled/'manifest.json', manifest)
        save_json(self.compiled/'compile.json', dict(artifact_kind=KIND, status='PARTIAL',
            manifest_sha256=sha(self.source/'manifest.json'), source_batch='../air',
            cases={self.name: dict(status='FAILED', error='shape failure')}))
        self.job = self.compiled/'models'/self.name/'omc'
        self.dumps = self.job/'compile-work/generated/dump_graph'
        self.dumps.mkdir(parents=True)
        (self.dumps/'02_after_infershape_1_fail.irpb').write_bytes(b'IR')
        save_json(self.job/'report.json', dict(stage='omg', returncode=1, error='shape failure'))
        (self.job/'compile.log').write_text('compilation failed\n')

    def tearDown(self):
        self.temp.cleanup()

    def inspect(self, **kwargs):
        with patch.object(inspection, 'Schema', FakeSchema), \
             patch.object(inspection, 'load_schema', return_value=FakeSchema()), \
             patch('subprocess.run', side_effect=AssertionError('must not run a compiler')):
            return inspection.inspect_compilation(self.compiled, self.root/'inspection', [self.name], ddk_root=self.root/'ddk', **kwargs)

    def test_observed_source_and_failed_ir_shapes_remain_distinct(self):
        before = {p: sha(p) for p in self.root.rglob('*') if p.is_file()}
        original_open = Path.open
        weight = self.weight
        def guarded(path, *args, **kwargs):
            if path == weight:
                raise AssertionError('external weights must not be read')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            result = self.inspect()
        self.assertEqual(result['status'], 'INSPECTED')
        row = result['cases'][self.name]
        source = row['source_graph']['graphs'][0]['nodes'][1]['output_desc'][0]
        inferred = row['ir_dumps'][0]['graphs'][0]['nodes'][1]['output_desc'][0]
        self.assertEqual(source['shape_evidence'], 'UNRESOLVED_EMPTY_OR_DYNAMIC_DESCRIPTOR')
        self.assertEqual(inferred['shape'], [256, 1024])
        self.assertEqual(row['golden']['shape'], [1, 256, 8192])
        self.assertEqual(row['compiler_report']['returncode'], 1)
        self.assertNotIn('PRIVATE_PAYLOAD', json.dumps(result))
        self.assertEqual(row['source_graph']['graphs'][0]['nodes'][0]['omitted_attributes'], ['value'])
        self.assertEqual({p: sha(p) for p in before}, before)
        self.assertEqual(result['device_execution'], 'NOT_RUN')

    def test_unknown_ir_schema_is_recorded_without_guessing_shapes(self):
        (self.dumps/'02_after_infershape_1_fail.irpb').write_bytes(b'UNKNOWN_SCHEMA')
        result = self.inspect()
        self.assertEqual(result['status'], 'PARTIAL')
        item = result['cases'][self.name]['ir_dumps'][0]
        self.assertEqual(item['status'], 'UNAVAILABLE')
        self.assertNotIn('graphs', item)

    def retain(self):
        from qwen35_moe_dflash.toolchain.compile import _retain_compiler_diagnostics
        evidence = _retain_compiler_diagnostics(self.dumps.parent, self.job, failure=True)
        for path in self.dumps.iterdir():
            path.unlink()
        save_json(self.job/'report.json', dict(stage='omg', returncode=1, compiler_diagnostics=evidence))
        return self.job/'diagnostics/02_after_infershape_1_fail.irpb'

    def test_reads_actual_retention_layout_without_scratch(self):
        path = self.retain()
        result = self.inspect()
        self.assertEqual(result['status'], 'INSPECTED')
        row = result['cases'][self.name]
        self.assertEqual(row['ir_inventory']['status'], 'FOUND')
        self.assertEqual(row['compiler_report']['compiler_diagnostics']['status'], 'RETAINED')
        self.assertEqual(row['ir_dumps'][0]['path'], str(path.relative_to(self.compiled)))
        self.assertEqual(row['ir_dumps'][0]['retention_binding'], 'SHA256_VERIFIED')
        self.assertEqual(row['ir_dumps'][0]['status'], 'INSPECTED')

    def test_changed_retained_ir_is_reported_without_decoding(self):
        path = self.retain()
        path.write_bytes(b'CHANGED')
        result = self.inspect()
        self.assertEqual(result['status'], 'PARTIAL')
        item = result['cases'][self.name]['ir_dumps'][0]
        self.assertEqual(item['status'], 'UNAVAILABLE')
        self.assertIn('fixture changed', item['error'])
        self.assertNotIn('graphs', item)

    def test_missing_and_budget_skipped_ir_are_explicitly_partial(self):
        self.retain().unlink()
        manifest = self.job/'diagnostics/manifest.json'
        record = json.loads(manifest.read_text())
        record['files'][0]['status'] = 'SKIPPED_SIZE_BUDGET'
        record['status'] = 'PARTIAL'
        save_json(manifest, record)
        result = self.inspect()
        self.assertEqual(result['status'], 'PARTIAL')
        row = result['cases'][self.name]
        self.assertEqual(row['ir_inventory']['status'], 'NOT_FOUND')
        self.assertEqual(row['ir_inventory']['skipped'][0]['status'], 'SKIPPED_SIZE_BUDGET')
        self.assertEqual(row['ir_dumps'], [])

    def test_absent_ir_reports_searched_paths(self):
        for path in self.dumps.iterdir():
            path.unlink()
        result = self.inspect()
        self.assertEqual(result['status'], 'PARTIAL')
        inventory = result['cases'][self.name]['ir_inventory']
        self.assertEqual(inventory['status'], 'NOT_FOUND')
        self.assertEqual(inventory['searched_directories'], ['diagnostics','compile-work/generated/dump_graph'])

    def test_retention_inventory_cannot_escape_case(self):
        self.retain()
        manifest = self.job/'diagnostics/manifest.json'
        record = json.loads(manifest.read_text())
        record['files'][0]['name'] = '../other.irpb'
        save_json(manifest, record)
        result = self.inspect()
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['cases'][self.name]['ir_inventory']['status'], 'UNAVAILABLE')

    def extensionless_dump(self):
        for path in self.dumps.iterdir():
            path.unlink()
        self.dumps.rmdir()
        self.dumps.write_bytes(b'IR')

    def test_extensionless_scratch_graph_is_inspected(self):
        self.extensionless_dump()
        result = self.inspect()
        self.assertEqual(result['status'], 'INSPECTED')
        row = result['cases'][self.name]
        self.assertEqual(row['ir_inventory']['dump_graph_path_kind'], 'FILE')
        self.assertEqual(row['ir_dumps'][0]['status'], 'INSPECTED')
        self.assertEqual(row['ir_dumps'][0]['path'], str(self.dumps.relative_to(self.compiled)))

    def test_compiler_retains_extensionless_dump_with_hash(self):
        from qwen35_moe_dflash.toolchain.compile import _retain_compiler_diagnostics
        self.extensionless_dump()
        kept = _retain_compiler_diagnostics(self.dumps.parent, self.job, failure=True)
        self.assertEqual(kept['status'], 'RETAINED')
        self.dumps.unlink()
        result = self.inspect()
        self.assertEqual(result['status'], 'INSPECTED')
        item = result['cases'][self.name]['ir_dumps'][0]
        self.assertTrue(item['path'].endswith('/diagnostics/dump_graph'))
        self.assertEqual(item['retention_binding'], 'SHA256_VERIFIED')

    def test_old_manifest_does_not_mask_unrecorded_extensionless_file(self):
        self.extensionless_dump()
        save_json(self.job/'diagnostics/manifest.json', dict(status='RETAINED', files=[]))
        result = self.inspect()
        self.assertEqual(result['status'], 'INSPECTED')
        self.assertEqual(result['cases'][self.name]['ir_dumps'][0]['retention_binding'], 'UNBOUND_LEGACY_FILE')

    def make_ddk_entries(self):
        ddk = self.root/'ddk'
        for relative, content in (
            ('tools/tools_omg/omg', b'#!/bin/sh\nexit 91\n'),
            ('tools/tools_omg/master/omg', b'\x7fELFhost fixture'),
            ('tools/tools_ascendc/set_ascendc_env.sh', b'export ASCEND_CUSTOM_OPP_PATH=/example/vendor\n'),
        ):
            path = ddk/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return ddk

    def test_entry_comparison_reads_scripts_and_only_declared_environment_keys(self):
        ddk = self.make_ddk_entries()
        before = {p:sha(p) for p in ddk.rglob('*') if p.is_file()}
        env = dict(PATH='/usr/bin', LD_LIBRARY_PATH='/old/lib', SOC_VERSION='original',
                   ASCEND_CUSTOM_OPP_PATH='/example/custom', UNRELATED_TOKEN='do-not-record')
        with patch('subprocess.run', side_effect=AssertionError('must not execute vendor code')):
            result = inspection.compiler_entry_metadata(ddk, inherited=env)
        self.assertEqual(result['status'], 'INSPECTED')
        self.assertEqual(result['wrapper_execution'], 'NOT_RUN')
        self.assertEqual(result['scripts']['tools/tools_omg/omg']['format'], 'TEXT')
        self.assertIn('exit 91', result['scripts']['tools/tools_omg/omg']['text'])
        self.assertEqual(result['probe_environment_changes']['SOC_VERSION'], dict(inherited='original',prepared='kirinx90'))
        self.assertEqual(result['probe_prepared_environment']['ASCEND_CUSTOM_OPP_PATH'], '/example/custom')
        self.assertNotIn('do-not-record', json.dumps(result))
        self.assertEqual(env['SOC_VERSION'], 'original')
        self.assertEqual({p:sha(p) for p in before}, before)

    def test_entry_inspection_includes_old_environment_without_claiming_missing_keys_absent(self):
        self.make_ddk_entries()
        save_json(self.job/'omg-environment.json', dict(environment={'SOC_VERSION':'kirinx90','UNRELATED_TOKEN':'secret'}))
        result = self.inspect(compiler_entry=True)
        self.assertEqual(result['status'], 'INSPECTED')
        prior = result['cases'][self.name]['recorded_compiler_environment']
        self.assertEqual(prior['environment'], {'SOC_VERSION':'kirinx90'})
        self.assertIn('ASCEND_CUSTOM_OPP_PATH', prior['unrecorded_keys'])
        self.assertNotIn('secret',json.dumps(result))

    def test_changed_source_manifest_is_rejected_before_writes(self):
        with (self.source/'manifest.json').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            self.inspect()
        self.assertFalse((self.root/'inspection').exists())

    def test_existing_diagnostics_are_not_overwritten(self):
        self.inspect()
        old = sha(self.root/'inspection/report.json')
        with self.assertRaises(FileExistsError):
            self.inspect()
        self.assertEqual(sha(self.root/'inspection/report.json'), old)

    def test_missing_source_schema_still_reports_compiler_failure(self):
        (self.folder/'air-schema.pb').unlink()
        result = self.inspect()
        row = result['cases'][self.name]
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertIn('source_error', row)
        self.assertEqual(row['compiler_report']['stage'], 'omg')
        self.assertEqual(row['ir_dumps'][0]['status'], 'INSPECTED')

    def test_unknown_cases_and_output_inside_batch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'select distinct cases'):
            inspection.inspect_compilation(self.compiled, self.root/'inspection', ['missing'])
        with self.assertRaisesRegex(ValueError, 'must be separate'):
            inspection.inspect_compilation(self.compiled, self.compiled/'inspection', [self.name])
        self.assertFalse((self.compiled/'inspection').exists())


if __name__ == '__main__':
    unittest.main()
