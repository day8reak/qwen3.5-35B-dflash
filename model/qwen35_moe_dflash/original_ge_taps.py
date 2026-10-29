"""Bind original-source input norms to an auxiliary feature-branch plan.

Sparse descriptors remain sparse. Expected dimensions are separate constraints
for later GE inference, never replacements for original graph metadata. This
module reads metadata only and neither creates AIR nor opens model weights.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re

from .original_ge_target import LAYER_IDS, _port, _proven_scalar_one, _unresolved_producer_shapes, audit_source


class _Pending(ValueError):
    def __init__(self, code, **evidence):
        self.code, self.evidence = code, evidence
        super().__init__(code)


def _need(condition, code, **evidence):
    if not condition:
        raise _Pending(code, **evidence)


def _float(desc, *, fp32=False):
    names = desc.get("schema_dtype_names", []) if isinstance(desc, dict) else []
    allowed = {"DT_FLOAT", "DT_FLOAT32"} if fp32 else {"DT_FLOAT", "DT_FLOAT32", "DT_FLOAT16"}
    return (isinstance(desc, dict) and desc.get("status") != "UNAVAILABLE" and type(desc.get("raw_dtype")) is int
            and bool(names) and set(names) <= allowed and desc.get("layout") == "ND")


def _same_type(left, right):
    return all(left.get(key) == right.get(key) for key in ("raw_dtype", "schema_dtype_names", "layout"))


def _shape_constraint(desc, expected, *, basis, port):
    _need(isinstance(desc, dict) and isinstance(desc.get("shape"), list), "DESCRIPTOR_SHAPE_MISSING", port=port, descriptor=desc)
    observed = desc["shape"]
    _need(observed == expected or observed == [], "EXPLICIT_SHAPE_CONTRADICTS_SOURCE_CONSTRAINT",
          port=port, descriptor=desc, expected_shape=expected)
    return {"port": port, "observed_shape": observed, "expected_shape": expected,
            "status": "EXPLICIT_SHAPE" if observed == expected else "SPARSE_DESCRIPTOR_REQUIRES_GE_INFERENCE",
            "basis": basis, "original_descriptor_changed": False}


class _Graph:
    def __init__(self, graph):
        nodes = graph.get("operations")
        _need(isinstance(nodes, list) and 1 <= len(nodes) <= 32768, "FULL_GRAPH_METADATA_REQUIRED")
        self.nodes, self.consumers, self.parents = {}, {}, {}
        for node in nodes:
            name = node.get("name") if isinstance(node, dict) else None
            _need(isinstance(name, str) and name and name not in self.nodes, "AMBIGUOUS_OPERATION_NAME", name=name)
            self.nodes[name] = node
        edges = 0
        for name, node in self.nodes.items():
            parents = []
            _need(isinstance(node.get("input", []), list), "INPUT_REFERENCES_UNAVAILABLE", node=name)
            for index, edge in enumerate(node.get("input", [])):
                if edge == "" or (isinstance(edge, str) and edge.endswith(":-1")):
                    continue  # Explicit unconnected optional/control slots are not data edges.
                try:
                    producer, port = _port(edge)
                except ValueError:
                    raise _Pending("UNRESOLVED_DATA_EDGE", node=name, input_index=index, reference=edge)
                target = self.nodes.get(producer)
                _need(target is not None and isinstance(target.get("output_desc"), list) and port < len(target["output_desc"]),
                      "UNRESOLVED_DATA_EDGE", node=name, input_index=index, reference=edge)
                parents.append(edge)
                self.consumers.setdefault(edge, []).append((name, index))
                edges += 1
                _need(edges <= 262144, "DATA_EDGE_BUDGET_EXCEEDED")
            self.parents[name] = parents
        # A source inference graph must be a DAG. Control edges were excluded.
        degrees = {name: len({ _port(edge)[0] for edge in refs }) for name, refs in self.parents.items()}
        children = {}
        for name, refs in self.parents.items():
            for parent in {_port(edge)[0] for edge in refs}:
                children.setdefault(parent, []).append(name)
        ready, seen = [name for name, count in degrees.items() if count == 0], 0
        while ready:
            name = ready.pop()
            seen += 1
            for child in children.get(name, []):
                degrees[child] -= 1
                if degrees[child] == 0:
                    ready.append(child)
        _need(seen == len(self.nodes), "CYCLIC_DATA_DEPENDENCIES")
        observed = dict(Counter(node.get("type") for node in nodes))
        _need(observed == graph.get("root_op_counts"), "OPERATION_TYPE_COUNTS_DIFFER_FROM_FULL_METADATA")
        self.type_counts = observed

    def output(self, reference):
        try:
            name, port = _port(reference)
        except ValueError:
            raise _Pending("UNRESOLVED_PRODUCER_PORT", reference=reference)
        node = self.nodes.get(name)
        _need(node is not None and port < len(node.get("output_desc", [])), "UNRESOLVED_PRODUCER_PORT", reference=reference)
        return node, node["output_desc"][port]

    def ancestors(self, reference):
        seen, todo = set(), [reference]
        while todo:
            edge = todo.pop()
            if edge in seen:
                continue
            seen.add(edge)
            _need(len(seen) <= 262144, "ANCESTOR_BUDGET_EXCEEDED")
            node, _ = self.output(edge)
            todo.extend(self.parents[node["name"]])
        return seen


def _anchor(index, graph, constants, template, sequence_length):
    path = template.format(layer=index)
    matches = [row for row in constants if row.get("file_path") == path]
    _need(len(matches) == 1, "EXACT_GAMMA_PATH_MISSING_OR_AMBIGUOUS", layer=index, path=path)
    metadata = matches[0]
    gamma = graph.nodes.get(metadata.get("name"))
    _need(gamma is not None and gamma.get("type") == "FileConstant" and len(gamma.get("output_desc", [])) == 1,
          "GAMMA_FILECONSTANT_PORT_UNRESOLVED", layer=index, path=path)
    gamma_desc = gamma["output_desc"][0]
    _need(metadata.get("outputs") == gamma["output_desc"] and _float(gamma_desc, fp32=True),
          "GAMMA_PROTO_DTYPE_OR_METADATA_DIFFERS", layer=index, descriptor=gamma_desc)
    _need(metadata.get("shape_attribute", {}).get("list_i") == [2048],
          "GAMMA_SHAPE_ATTRIBUTE_REQUIRED", layer=index, shape_attribute=metadata.get("shape_attribute"))
    uses = graph.consumers.get(gamma["name"] + ":0", [])
    _need(len(uses) == 1 and uses[0][1] == 0, "GAMMA_ADD_CONSUMER_NOT_UNIQUE_INPUT_ZERO", layer=index, consumers=uses)
    add = graph.nodes[uses[0][0]]
    ins, descs, outs = add.get("input", []), add.get("input_desc", []), add.get("output_desc", [])
    _need(add.get("type") == "Add" and len(ins) == len(descs) == 2 and len(outs) == 1,
          "GAMMA_ADD_INTERFACE_UNRESOLVED", layer=index, node=add["name"])
    _need(descs[0].get("shape") == [2048] and _float(descs[0], fp32=True) and _same_type(gamma_desc, descs[0]),
          "GAMMA_CONSUMER_VECTOR_DECLARATION_REQUIRED", layer=index, descriptor=descs[0])
    scalar, scalar_desc = graph.output(ins[1])
    _need(ins[1].endswith(":0") and descs[1] == scalar_desc and _float(scalar_desc, fp32=True)
          and _proven_scalar_one(scalar, gamma_desc),
          "EXACT_FP32_SCALAR_ONE_NOT_PROVEN", layer=index, constant=ins[1])
    _need(_float(outs[0], fp32=True) and _same_type(outs[0], gamma_desc), "GAMMA_ADD_OUTPUT_DTYPE_DIFFERS", layer=index)
    gamma_shape = _shape_constraint(gamma_desc, [2048], basis="FileConstant shape attribute and independent Add input0 both declare [2048]", port=gamma["name"] + ":0")
    add_shape = _shape_constraint(outs[0], [2048], basis="Explicit vector[2048] + exact scalar-one Add operands", port=add["name"] + ":0")
    uses = graph.consumers.get(add["name"] + ":0", [])
    _need(len(uses) == 1 and uses[0][1] == 1, "GAMMA_ADD_OUTPUT_NOT_UNIQUE_NORM_GAMMA", layer=index, consumers=uses)
    norm = graph.nodes[uses[0][0]]
    norm_ins, norm_descs = norm.get("input", []), norm.get("input_desc", [])
    _need(norm.get("type") == "RmsNorm" and len(norm_ins) == len(norm_descs) == 2 and norm.get("output_desc"),
          "INPUT_NORM_INTERFACE_UNRESOLVED", layer=index, node=norm["name"])
    _need(_float(norm_descs[1], fp32=True) and _same_type(norm_descs[1], outs[0]), "NORM_GAMMA_DTYPE_DIFFERS", layer=index)
    norm_gamma_shape = _shape_constraint(norm_descs[1], [2048], basis="Exact producer is verified vector-plus-one Add", port=norm["name"] + ":input1")
    expected = [1, sequence_length, 2048]
    constraints = [gamma_shape, add_shape, norm_gamma_shape,
        _shape_constraint(norm_descs[0], expected, basis="Audited source input norm preserves original embedding T and hidden width", port=norm["name"] + ":input0")]
    _need(_float(norm_descs[0], fp32=True) and _float(norm["output_desc"][0], fp32=True),
          "INPUT_NORM_FP32_PORT_EVIDENCE_REQUIRED", layer=index)
    constraints.append(_shape_constraint(norm["output_desc"][0], expected,
        basis="Audited npu_rms_norm output0 preserves data-input shape", port=norm["name"] + ":0"))
    return {"layer_index": index, "gamma_path": path, "gamma_reference": gamma["name"] + ":0",
        "gamma_proto_descriptor": deepcopy(gamma_desc), "gamma_dtype_attribute": metadata.get("dtype_attribute"),
        "dtype_attribute_scope": "Recorded only; GE attribute datatype IDs are not protobuf dtype IDs",
        "gamma_add": add["name"], "scalar_one_reference": ins[1], "scalar_one_evidence": scalar.get("scalar_constant"),
        "norm": norm["name"], "norm_output0_consumers": len(graph.consumers.get(norm["name"] + ":0", [])),
        "norm_data_reference": norm_ins[0], "shape_constraints": constraints}


def _source_hidden(anchor, graph, sequence_length):
    index, reference = anchor["layer_index"], anchor["norm_data_reference"]
    node, desc = graph.output(reference)
    _need(reference.endswith(":0") and _float(desc, fp32=True), "NORM_DATA_FP32_PRODUCER_REQUIRED", layer=index, reference=reference)
    expected = [1, sequence_length, 2048]
    constraints = [_shape_constraint(desc, expected, basis="Source input-norm hidden constraint", port=reference)]
    cast = None
    if node.get("type") == "Cast":
        ins, inputs, outputs = node.get("input", []), node.get("input_desc", []), node.get("output_desc", [])
        consumers = graph.consumers.get(reference, [])
        norm_use = (anchor["norm"], 0)
        _need(len(ins) == len(inputs) == len(outputs) == 1 and consumers.count(norm_use) == 1,
              "PENDING_CAST_LINEAGE", layer=index, cast=node["name"], consumers=graph.consumers.get(reference))
        residual_uses = [use for use in consumers if use != norm_use]
        # Original Cast_1 is both the norm input and the residual stream. A
        # second consumer is allowed only when the exact same value is Add
        # input0 and its other operand descends from this norm output0.
        _need(len(residual_uses) <= 1, "PENDING_CAST_LINEAGE", layer=index, cast=node["name"], consumers=consumers)
        for consumer_name, port in residual_uses:
            consumer = graph.nodes[consumer_name]
            other = consumer.get("input", [])
            _need(consumer.get("type") == "Add" and port == 0 and len(other) == 2
                  and anchor["norm"] + ":0" in graph.ancestors(other[1]),
                  "PENDING_CAST_LINEAGE", layer=index, cast=node["name"], consumer=consumer_name, input_port=port)
        source, source_desc = graph.output(ins[0])
        _need(_float(source_desc) and _float(inputs[0]) and _same_type(inputs[0], source_desc)
              and _float(outputs[0], fp32=True), "PENDING_CAST_LINEAGE", layer=index, cast=node["name"])
        constraints.extend([_shape_constraint(inputs[0], expected, basis="Shape-preserving Cast input", port=node["name"] + ":input0"),
                            _shape_constraint(source_desc, expected, basis="Same explicit value feeds the Cast and residual stream", port=ins[0])])
        cast = {"reference": reference, "source_reference": ins[0], "input_descriptor": deepcopy(inputs[0]),
                "output_descriptor": deepcopy(outputs[0]), "residual_consumer_edges": residual_uses}
        if index:
            reference, node, desc = ins[0], source, source_desc
    if index:
        _need(reference.endswith(":0") and node.get("type") == "Add" and len(node.get("input", [])) == 2,
              "PREVIOUS_LAYER_OUTPUT_ADD_UNRESOLVED", layer=index, reference=reference, actual_type=node.get("type"))
    return {"producer_reference": reference, "descriptor": deepcopy(desc), "cast_lineage": cast,
            "shape_constraints": constraints}


def plan_original_ge_taps(air_report, source_audit, gamma_template, *, sequence_length):
    """Return 40 bound anchors and eight typed feature plans, or exact gaps."""
    result = {"schema_version": 1, "artifact_kind": "qwen35-original-ge-tap-plan", "status": "PENDING_GRAPH_BINDING",
        "source_to_air_revision": "UNVERIFIED", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
        "numerical_validation": "NOT_RUN", "original_descriptors_changed": False, "taps": [], "pending": []}
    try:
        _need(type(sequence_length) is int and 1 <= sequence_length <= 2048, "EXPLICIT_SOURCE_SEQUENCE_LENGTH_REQUIRED")
        _need(isinstance(gamma_template, str) and gamma_template.count("{layer}") == 1
              and "{" not in gamma_template.replace("{layer}", "") and "}" not in gamma_template.replace("{layer}", ""),
              "EXPLICIT_EXACT_GAMMA_TEMPLATE_REQUIRED")
        _need(isinstance(source_audit, dict) and source_audit.get("status") == "MATCHED_AUDITED_SOURCE_RECEIPTS", "SOURCE_AUDIT_REQUIRED")
        verified_source = audit_source({"files": source_audit.get("files")})
        _need(verified_source["status"] == "MATCHED_AUDITED_SOURCE_RECEIPTS", "SOURCE_AUDIT_REQUIRED")
        _need(air_report.get("artifact_kind") == "qwen35-original-air-inspection"
              and air_report.get("source_preservation") == "UNCHANGED_BY_STAT"
              and re.fullmatch(r"[0-9a-f]{64}", str(air_report.get("air", {}).get("sha256", ""))), "STRICT_AIR_IDENTITY_REQUIRED")
        candidates = [row for row in air_report.get("candidates", []) if row.get("format") in {"RAW_MODELDEF", "RAW_GRAPHDEF"}
                      and row.get("protobuf_decode") == "PASSED" and row.get("metadata_inspection") == "COMPLETE" and isinstance(row.get("graph"), dict)]
        _need(len(candidates) == 1, "UNAMBIGUOUS_COMPLETE_RAW_GRAPH_REQUIRED")
        metadata = candidates[0]["graph"]
        boundary = metadata.get("original_boundary_check", {})
        shape_observation = _unresolved_producer_shapes(metadata)
        _need(boundary.get("status") == "MATCHED_COUNTS_SHAPES_DTYPES_ONLY" or shape_observation is not None,
              "PENDING_ORIGINAL_ABI_BINDING", original_boundary_check=boundary)
        graph = _Graph(metadata)
        inputs = {row["name"]: row for row in metadata.get("inputs", [])}
        embedding, scale = inputs.get("arg1_1", {}), inputs.get("arg0_1", {})
        _need(embedding.get("type") == "Data" and embedding.get("shape") == [1, sequence_length, 2048]
              and embedding.get("schema_dtype_names") == ["DT_INT8"] and embedding.get("layout") == "ND"
              and scale.get("type") == "Data" and scale.get("shape") == [1, sequence_length, 1] and _float(scale, fp32=True),
              "ORIGINAL_EMBEDDING_SEQUENCE_ABI_UNRESOLVED", embedding=embedding, scale=scale)
        constants = metadata.get("file_constants", {})
        _need(constants.get("count") == constants.get("shown") and isinstance(constants.get("examples"), list), "FULL_FILECONSTANT_METADATA_REQUIRED")
        anchors = []
        for index in range(40):
            try:
                anchors.append(_anchor(index, graph, constants["examples"], gamma_template, sequence_length))
            except _Pending as error:
                result["pending"].append({"code": error.code, **error.evidence})
        result["input_norm_anchors"] = anchors
        _need(len(anchors) == 40 and len({a["norm"] for a in anchors}) == 40, "FORTY_DISTINCT_INPUT_NORM_ANCHORS_REQUIRED")
        hidden = [_source_hidden(anchor, graph, sequence_length) for anchor in anchors]
        norm_refs = {a["norm"] + ":0": a["layer_index"] for a in anchors}
        ordered = []
        for index, value in enumerate(hidden):
            ancestors = graph.ancestors(value["producer_reference"])
            observed = sorted(layer for ref, layer in norm_refs.items() if ref in ancestors)
            _need(observed == list(range(index)), "ORDERED_INPUT_NORM_DEPENDENCIES_UNRESOLVED", layer=index, observed_predecessors=observed)
            if index == 0:
                _need({"arg1_1:0", "arg0_1:0"} <= ancestors, "ORIGINAL_EMBEDDING_DATAFLOW_UNRESOLVED")
            ordered.append({"input_norm_layer": index, "producer_reference": value["producer_reference"],
                            "predecessor_input_norm_layers": observed})
        outputs = metadata.get("outputs", [])
        _need(len(outputs) == 81, "ORIGINAL_81_ORDERED_OUTPUTS_REQUIRED")
        logits_ref = outputs[0].get("producer_reference")
        logits_ancestors = graph.ancestors(logits_ref)
        _need({ref for ref in norm_refs} <= logits_ancestors, "ALL_FORTY_INPUT_NORMS_MUST_FEED_ORIGINAL_LOGITS")
        taps, casts = [], []
        for position, layer in enumerate(LAYER_IDS):
            value = hidden[layer + 1]
            name = f"__dflash_features/cast_{position}"
            taps.append({"post_decoder_layer_index": layer, "next_input_norm_layer_index": layer + 1,
                "producer_reference": value["producer_reference"], "descriptor": value["descriptor"],
                "expected_shape": [1, sequence_length, 2048], "feature_dtype": "float16",
                "cast_lineage": value["cast_lineage"], "shape_constraints": value["shape_constraints"]})
            casts.append({"name": name, "type": "Cast", "input": value["producer_reference"],
                "input_descriptor": value["descriptor"], "output_descriptor": {"shape": [1, sequence_length, 2048], "dtype": "float16", "layout": "ND"},
                "dtype_symbol": "ge::DT_FLOAT16", "shape_inference_required": True})
        result.update(status="FEATURE_BRANCH_PLAN_REVIEW_REQUIRED", feature_plan_available=True,
            air_identity=deepcopy(air_report["air"]), schema_identity=deepcopy(air_report.get("schema", {})),
            sequence_length=sequence_length, original_input_count=89, original_output_count=81,
            original_boundary_check=deepcopy(boundary), producer_shape_observation=shape_observation,
            actual_type_counts=graph.type_counts, bound_input_norm_anchor_count=40,
            ordered_dependencies=ordered, original_logits_dependency="ALL_40_INPUT_NORM_OUTPUT0_PORTS_OBSERVED",
            taps=taps, feature_branch={"casts": casts, "concat": {"name": "__dflash_features/concat", "type": "ConcatD", "axis": 2, "N": 8,
                "input": [node["name"] + ":0" for node in casts], "output_descriptor": {"shape": [1, sequence_length, 16384], "dtype": "float16", "layout": "ND"}},
                "preserved_output_references": [row["producer_reference"] for row in outputs], "feature_output_index": 81,
                "requires_selected_GE_shape_inference_and_roundtrip": True},
            scope="Exact source-policy gamma/one/norm and ordered dataflow candidates; expected shapes are new-branch constraints only. No original shape filling, source revision equivalence, ABI/device/accuracy pass.")
    except _Pending as error:
        result["status"] = error.code if error.code.startswith("PENDING_") else "PENDING_GRAPH_BINDING"
        result["pending"].append({"code": error.code, **error.evidence})
    except (ValueError, TypeError, KeyError) as error:
        result["pending"].append({"code": "UNSUPPORTED_METADATA", "error": str(error)[:256]})
    return result
