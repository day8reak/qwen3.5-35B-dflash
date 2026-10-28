"""Explicit installation-supplied quantized GE contracts; no inferred ABI.

This local adapter has no built-in vendor port table. A complete contract,
including unconnected optional slots and dynamic list registration, is required.
Its presence proves only the caller's declaration, never target support.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .recipe import _attribute


_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ROLES = {
    "dynamic_quant": ({"x": "float16"}, {"quantized": "int8", "scale": "float32"}, set()),
    "quant_matmul": ({"x": "int8", "weight": "int32", "scale": "float32",
                       "token_scale": "float32"}, {"y": "float16"}, set()),
    "grouped_matmul": ({"x": "int8", "weight": "int32", "scale": "int64",
                         "token_scale": "float32", "group_list": "int64"},
                        {"y": "float16"}, {"group_list_type", "split_item", "group_type", "act_type"}),
}


def _schema(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("complete ordered GE port schema is required")
    names, connected, declared = [], [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"name", "kind", "count"}:
            raise ValueError("schema entries require name, kind and optional dynamic count")
        name, kind = row.get("name"), row.get("kind")
        if not isinstance(name, str) or not _NAME.fullmatch(name) or kind not in {"required", "optional", "dynamic"}:
            raise ValueError("invalid GE schema port")
        if name in declared:
            raise ValueError("duplicate declared GE schema port")
        declared.add(name)
        if kind == "dynamic":
            count = row.get("count")
            if type(count) is not int or not 0 <= count <= 1:
                raise ValueError("quant adapter supports explicitly declared dynamic lists of zero or one")
            expanded = [name + str(i) for i in range(count)]
        else:
            if "count" in row:
                raise ValueError("only dynamic schema ports have count")
            expanded = [name]
        if any(port in names for port in expanded):
            raise ValueError("duplicate expanded GE schema port")
        names.extend(expanded)
        if kind != "optional":
            connected.extend(expanded)
    return names, connected


class QuantOperatorBindings:
    def __init__(self, document):
        doc = deepcopy(document)
        if not isinstance(doc, dict) or doc.get("schema_version") != 1 or doc.get("target") != "kirinx90":
            raise ValueError("quant binding requires schema_version=1 and target=kirinx90")
        if not isinstance(doc.get("ddk_id"), str) or not doc["ddk_id"].strip():
            raise ValueError("quant binding requires a selected DDK identity")
        source = doc.get("source", {})
        if (not isinstance(source, dict) or not isinstance(source.get("path"), str) or not source["path"]
                or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")))):
            raise ValueError("quant binding requires explicit source path and SHA256")
        operators = doc.get("operators")
        if not isinstance(operators, dict) or set(operators) != set(_ROLES):
            raise ValueError("quant bindings require dynamic_quant, quant_matmul and grouped_matmul")
        schemas = {}
        for role, (ins, outs, attrs) in _ROLES.items():
            spec = operators[role]
            typ = spec.get("op_type")
            if not isinstance(typ, str) or not _NAME.fullmatch(typ):
                raise ValueError("quant op_type must be an explicit GE identifier")
            if spec.get("descriptor_format") != "ND":
                raise ValueError("quant adapter currently requires explicit ND carrier descriptors")
            for direction, expected in (("inputs", ins), ("outputs", outs)):
                ports = spec.get(direction, {})
                if not isinstance(ports, dict) or set(ports) != set(expected):
                    raise ValueError(f"{role} {direction} must bind {sorted(expected)}")
                names, required = _schema(spec.get(direction[:-1] + "_schema"))
                actual = []
                for logical, port in ports.items():
                    dtype = expected[logical]
                    if not isinstance(port, dict) or set(port) != {"port", "dtype"} or port["dtype"] != dtype:
                        raise ValueError(f"{role}.{logical} requires explicit {dtype} port")
                    actual.append(port["port"])
                if len(set(actual)) != len(actual) or not set(actual) <= set(names) or not set(required) <= set(actual):
                    raise ValueError(f"{role} {direction} does not match its complete schema")
                if direction == "outputs" and actual != names:
                    raise ValueError("quant output bindings must retain full registered ordinal order")
                if direction == "outputs" and any(row["kind"] == "optional" for row in spec["output_schema"]):
                    raise ValueError("quant outputs require required or dynamic GE registration")
            mappings = spec.get("attrs", {})
            if not isinstance(mappings, dict) or set(mappings) != attrs:
                raise ValueError(f"{role} requires explicit attribute mappings {sorted(attrs)}")
            for value in mappings.values():
                if (not isinstance(value, dict) or set(value) != {"name", "type"}
                        or not _NAME.fullmatch(str(value["name"])) or value["type"] != "int"):
                    raise ValueError("quant logical attributes require explicit integer GE mappings")
            if len({value["name"] for value in mappings.values()}) != len(mappings):
                raise ValueError("quant logical attributes must have unique GE names")
            static = spec.get("static_attrs", {})
            if not isinstance(static, dict) or set(static) & {value["name"] for value in mappings.values()}:
                raise ValueError("quant static attributes cannot override mapped attributes")
            for name, value in static.items():
                if not _NAME.fullmatch(name):
                    raise ValueError("invalid static GE attribute name")
                _attribute(value)
            if role != "dynamic_quant":
                if spec.get("weight_encoding") != "I4ZN_INT32" or spec.get("group_size") != 256:
                    raise ValueError("quant matmul contract must explicitly declare I4ZN_INT32 group_size=256")
                wanted = "FP32_GROUP_SCALE" if role == "quant_matmul" else "VENDOR_QPARAM_INT64"
                if spec.get("scale_encoding") != wanted:
                    raise ValueError(f"{role} requires {wanted}")
            pair = (spec["input_schema"], spec["output_schema"])
            if typ in schemas and schemas[typ] != pair:
                raise ValueError("one GE type cannot have conflicting port schemas")
            schemas[typ] = pair
        self._document = doc

    @classmethod
    def from_file(cls, path):
        payload = Path(path).read_bytes()
        if len(payload) > 1 << 20:
            raise ValueError("quant binding exceeds 1 MiB")

        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate quant binding JSON key: " + key)
                result[key] = value
            return result

        result = cls(json.loads(payload, object_pairs_hook=unique_pairs))
        result.sha256 = hashlib.sha256(payload).hexdigest()
        return result

    def manifest(self):
        return deepcopy(self._document)

    def call(self, graph, role, *, inputs, outputs, attrs=None, name):
        spec = self._document["operators"][role]
        expected_inputs, expected_outputs, logical_attrs = _ROLES[role]
        attrs = {} if attrs is None else attrs
        if set(inputs) != set(expected_inputs) or set(outputs) != set(expected_outputs) or set(attrs) != logical_attrs:
            raise ValueError("quant invocation does not match declared logical contract")
        if any(inputs[key].dtype != dtype for key, dtype in expected_inputs.items()):
            raise ValueError("quant invocation input dtype differs from its contract")
        if any(outputs[key][1] != dtype for key, dtype in expected_outputs.items()):
            raise ValueError("quant invocation output dtype differs from its contract")
        bound = {spec["inputs"][key]["port"]: value for key, value in inputs.items()}
        ordered = {port: bound[port] for port in _schema(spec["input_schema"])[0] if port in bound}
        actual_attrs = deepcopy(spec.get("static_attrs", {}))
        for logical, value in attrs.items():
            field = spec["attrs"][logical]
            actual_attrs[field["name"]] = {"type": field["type"], "value": value}
        values = graph.custom(spec["op_type"], ordered,
            {desc["port"]: outputs[logical] for logical, desc in spec["outputs"].items()}, actual_attrs, name)
        return {logical: values[desc["port"]] for logical, desc in spec["outputs"].items()}

    def extend_recipe_cpp(self, source):
        """Register declared full schemas; verify connected ports at true indices."""
        unique = {spec["op_type"]: spec for spec in self._document["operators"].values()}
        input_branches, output_branches, index_branches, count_branches = [], [], [], []
        for typ, spec in unique.items():
            quoted = json.dumps(typ)
            for direction, branches in (("input", input_branches), ("output", output_branches)):
                calls = []
                for row in spec[direction + "_schema"]:
                    method = ("Dynamic" if row["kind"] == "dynamic" else "Optional" if row["kind"] == "optional" else "")
                    method += direction.capitalize() + "Register"
                    args = json.dumps(row["name"]) + (", " + str(row["count"]) if row["kind"] == "dynamic" else "")
                    calls.append(f"{method}({args});")
                branches.append(f"if (r.type == {quoted}) {{ {' '.join(calls)} }} else ")
            names = _schema(spec["input_schema"])[0]
            table = ",".join(json.dumps(name) for name in names)
            index_branches.append(f"if (r.type == {quoted}) {{ const std::vector<std::string> names = {{{table}}}; "
                "auto found = std::find(names.begin(), names.end(), port.name); "
                'Require(found != names.end(), "unknown quant schema port"); '
                "return static_cast<int32_t>(std::distance(names.begin(), found)); }")
            count_branches.append(f"if (r.type == {quoted}) return {len(names)};")
        replacements = [
            ('    if (r.type == "AdnFusedInferAttention") {', "    " + "".join(input_branches) + 'if (r.type == "AdnFusedInferAttention") {'),
            ('    for (const auto& p : r.outputs) OutputRegister(p.name.c_str());',
             "    " + "".join(output_branches) + 'for (const auto& p : r.outputs) OutputRegister(p.name.c_str());'),
            ('  if (r.type != "AdnFusedInferAttention") return static_cast<int32_t>(compact);',
             "  " + "\n  ".join(index_branches) + '\n  if (r.type != "AdnFusedInferAttention") return static_cast<int32_t>(compact);'),
            ('int32_t ExpectedInputIndex(const Record& r, const Port& port, size_t compact) {',
             'size_t QuantInputCount(const Record& r, size_t fallback) { '
             + " ".join(count_branches) + ' return fallback; }\n'
             + 'int32_t ExpectedInputIndex(const Record& r, const Port& port, size_t compact) {'),
            ('node.GetInputsSize() == (r.type == "Data" ? 1 : r.inputs.size())',
             'node.GetInputsSize() == QuantInputCount(r, r.type == "Data" ? 1 : r.inputs.size())'),
        ]
        for old, new in replacements:
            if source.count(old) != 1:
                raise RuntimeError("GE recipe serializer changed; quant schema extension requires review")
            source = source.replace(old, new)
        return source
