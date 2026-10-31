"""Prove FP32 producer identity across explicitly owned CPUCL partitions.

This supports the observed packed Data/NetOutput interface: one Data operation
per CPU GraphDef exposes the ordered GraphOp inputs, and one NetOutput exposes
the ordered outputs. Complete interface port counts must agree; descriptors are
checked on every selected FP32 path port. No node-name
guessing, shape-only producer matching, Cast traversal or opaque NPU traversal is
accepted. Binary task ownership is established by the caller before construction.
"""
from __future__ import annotations

import json
import re


class PartitionInterfaceError(ValueError):
    """A rejected packed interface with exact, serializable mismatch evidence."""

    def __init__(self, reason, details):
        super().__init__(reason)
        self.details = details


def _interface_error(reason, owner, *, node=None, main_direction=None,
                     child_direction=None, port=None, expected=None, actual=None,
                     differing_fields=(), path_port=None):
    relation = "UNKNOWN"
    if path_port is not None:
        relation = ("PATH_PORT" if (main_direction, port) == path_port
                    else "OTHER_INTERFACE_PORT")
    return PartitionInterfaceError(reason, {
        "owner": owner, "boundary_type": node.get("type") if node else None,
        "node": node.get("name") if node else None,
        "main_direction": main_direction, "child_direction": child_direction,
        "port": port, "expected": expected, "actual": actual,
        "differing_fields": list(differing_fields), "relation_to_path": relation})


def _index(value):
    if type(value) is not int or value < 0:
        raise ValueError("partition port must be a nonnegative integer")
    return value


def _ops(graph):
    nodes = graph.get("op")
    if not isinstance(nodes, list) or not 0 < len(nodes) <= 8192:
        raise ValueError("partition operation list is missing or exceeds its bound")
    result = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("name"), str) or not node["name"]:
            raise ValueError("partition operation has no name")
        if node["name"] in result:
            raise ValueError("partition operation name is ambiguous")
        result[node["name"]] = node
    return result


def _edge(node, port):
    count = len(node.get("input_desc", []))
    if "input" in node:
        refs = node["input"]
        if not isinstance(refs, list):
            raise ValueError("partition input references are not a list")
        edges = []
        for ref in refs:
            if not isinstance(ref, str) or ":" not in ref:
                raise ValueError("partition input reference is unresolved")
            name, raw = ref.rsplit(":", 1)
            if not name or not re.fullmatch(r"-?\d+", raw):
                raise ValueError("partition input reference is malformed")
            output = int(raw)
            if output >= 0:
                edges.append((name, output))
            elif output != -1:
                raise ValueError("partition control port is unsupported")
        if len(edges) != count or port >= count:
            raise ValueError("partition input reference count differs from descriptors")
        return edges[port]
    names, indices = node.get("src_name"), node.get("src_index")
    if (not isinstance(names, list) or not isinstance(indices, list)
            or len(names) != count or len(indices) != count or port >= count
            or not isinstance(names[port], str) or not names[port]):
        raise ValueError("partition input producer metadata is unresolved")
    return names[port], _index(indices[port])


class PartitionEdgeResolver:
    """Resolve a required producer through validated CPU partition interfaces.

    ``observe_descriptor(desc, enum_fields)`` must return schema-resolved dtype
    and shape. Interfaces are checked lazily, so adding this context does not
    change existing direct, same-partition precision checks.
    """

    def __init__(self, main_graph, cpu_graphs, observe_descriptor):
        self.main = _ops(main_graph)
        self.cpu = {}
        self.observe = observe_descriptor
        for item in cpu_graphs:
            owner = item.get("owner")
            if not isinstance(owner, str) or not owner or owner in self.cpu:
                raise ValueError("CPU partition ownership is missing or ambiguous")
            graph = item.get("graph")
            if (item.get("cl_name") != "CPUCL" or not isinstance(graph, dict)
                    or graph.get("name") != owner):
                raise ValueError("CPU partition ownership is inconsistent")
            self.cpu[owner] = (item, _ops(graph))

    def _descriptor(self, node, direction, port, enums):
        descs = node.get(direction)
        if not isinstance(descs, list) or port >= len(descs):
            raise ValueError("partition descriptor port is missing")
        observed = self.observe(descs[port], enums)
        shape = observed.get("shape")
        if (observed.get("dtype") not in {"float32", "float16", "int32", "int64", "bool"}
                or not isinstance(shape, list) or len(shape) > 16
                or any(type(dim) is not int or dim < 0 for dim in shape)):
            raise ValueError("partition descriptor dtype/shape is unresolved")
        return {"dtype": observed["dtype"], "shape": shape,
                "layout": descs[port].get("layout")}

    def _interface(self, owner, *, path_port=None):
        if owner not in self.cpu:
            raise _interface_error("upstream partition has no explicit CPUCL GraphDef ownership", owner,
                                   expected="CPUCL GraphDef", actual=None,
                                   differing_fields=("owner",), path_port=path_port)
        item, nodes = self.cpu[owner]
        main = self.main.get(owner)
        if not isinstance(main, dict) or main.get("type") != "GraphOp":
            raise _interface_error("CPU GraphDef has no matching main GraphOp owner", owner,
                                   expected="GraphOp", actual=main.get("type") if isinstance(main, dict) else None,
                                   differing_fields=("type",), path_port=path_port)
        attrs = main.get("attr", [])
        if not isinstance(attrs, list):
            raise _interface_error("main GraphOp ownership attributes are invalid", owner,
                                   expected="attribute list", actual=type(attrs).__name__,
                                   differing_fields=("attr",), path_port=path_port)
        libraries = []
        for row in attrs:
            if not isinstance(row, dict) or row.get("key") != "cl_name":
                continue
            value = row.get("value")
            if not isinstance(value, dict):
                raise _interface_error("main GraphOp CPUCL ownership attribute is malformed", owner,
                                       expected="attribute object", actual=type(value).__name__,
                                       differing_fields=("cl_name",), path_port=path_port)
            libraries.append(value.get("s"))
        if libraries != ["CPUCL"]:
            raise _interface_error("main GraphOp does not have unique CPUCL ownership", owner,
                                   expected=["CPUCL"], actual=libraries,
                                   differing_fields=("cl_name",), path_port=path_port)
        data = [node for node in nodes.values() if node.get("type") == "Data"]
        netout = [node for node in nodes.values() if node.get("type") == "NetOutput"]
        if len(data) != 1 or len(netout) != 1:
            raise _interface_error("CPU packed interface requires one Data and one NetOutput", owner,
                                   expected={"Data": 1, "NetOutput": 1},
                                   actual={"Data": len(data), "NetOutput": len(netout)},
                                   differing_fields=("boundary_count",), path_port=path_port)
        enums = item["descriptor_enum_fields"]
        for child, main_key in ((data[0], "input_desc"), (netout[0], "output_desc")):
            parent_descs = main.get(main_key)
            if not isinstance(parent_descs, list) or len(parent_descs) > 8192:
                raise _interface_error("main GraphOp interface descriptors are missing or exceed bound", owner,
                                       node=child, main_direction=main_key,
                                       expected="descriptor list of at most 8192 ports",
                                       actual=len(parent_descs) if isinstance(parent_descs, list) else None,
                                       differing_fields=("port_count",), path_port=path_port)
            count = len(parent_descs)
            for key in ("input_desc", "output_desc"):
                if not isinstance(child.get(key), list) or len(child[key]) != count:
                    raise _interface_error("CPU packed interface port counts differ from its GraphOp", owner,
                                           node=child, main_direction=main_key, child_direction=key,
                                           expected=count, actual=len(child[key]) if isinstance(child.get(key), list) else None,
                                           differing_fields=("port_count",), path_port=path_port)
        return main, data[0], netout[0], enums, nodes

    def _require_path_descriptor(self, owner, node, direction, port, enums, wanted, *, reason):
        """Validate one path port, retaining both descriptors on any failure."""
        main_direction = ("input_desc" if node.get("type") == "Data" else "output_desc"
                          if node.get("type") == "NetOutput" else direction)
        child_direction = None if node.get("type") == "GraphOp" else direction
        try:
            actual = self._descriptor(node, direction, port, enums)
        except ValueError as error:
            descs = node.get(direction)
            raw = descs[port] if isinstance(descs, list) and port < len(descs) else None
            raise _interface_error(str(error), owner, node=node, main_direction=main_direction,
                                   child_direction=child_direction, port=port,
                                   expected=wanted, actual=raw, differing_fields=("descriptor",),
                                   path_port=(main_direction, port)) from error
        if actual != wanted:
            raise _interface_error(reason, owner, node=node, main_direction=main_direction,
                                   child_direction=child_direction, port=port, expected=wanted, actual=actual,
                                   differing_fields=[field for field in ("dtype", "shape", "layout")
                                                     if wanted[field] != actual[field]],
                                   path_port=(main_direction, port))
        return actual

    def resolve(self, owner, producer, output_port, expected):
        """Return MATCHED only after reaching the exact declared source output."""
        path, visited = [], set()
        try:
            output_port = _index(output_port)
            expected_port = _index(expected["producer_output_index"])
            wanted = {"dtype": expected["dtype"], "shape": expected["shape"], "layout": "ND"}
            if wanted["dtype"] != "float32":
                raise ValueError("cross-partition proof only supports FP32 producer edges")
            for _ in range(128):
                state = (owner, producer, output_port)
                if state in visited:
                    raise ValueError("CPU partition producer path contains a cycle")
                visited.add(state)
                current = self.cpu.get(owner, (None, {}))[1].get(producer)
                interface_port = (("input_desc", output_port)
                                  if isinstance(current, dict) and current.get("type") == "Data" else None)
                main, data, netout, enums, nodes = self._interface(owner, path_port=interface_port)
                node = nodes.get(producer)
                if not isinstance(node, dict):
                    raise ValueError("CPU partition producer is unresolved")
                descriptor = self._require_path_descriptor(
                    owner, node, "output_desc", output_port, enums, wanted,
                    reason="producer path changes FP32 dtype, shape or layout")
                path.append({"kind": "CPU_OUTPUT", "owner": owner, "node": producer,
                             "port": output_port, "descriptor": descriptor})
                if producer == expected["producer"] and output_port == expected_port:
                    if node.get("type") in {"Data", "NetOutput", "GraphOp", "Cast", "CastT"}:
                        raise ValueError("declared arithmetic producer resolves to an interface or Cast")
                    return {"status": "MATCHED", "owner": owner, "producer": producer,
                            "output_port": output_port, "path": path}
                if node is not data:
                    raise ValueError("cross-partition path reaches an undeclared operation or Cast")
                self._require_path_descriptor(owner, data, "input_desc", output_port, enums, wanted,
                                              reason="CPU Data input changes FP32 dtype, shape or layout")
                self._require_path_descriptor(owner, main, "input_desc", output_port, enums, wanted,
                                              reason="main GraphOp input changes FP32 dtype, shape or layout")
                parent_name, parent_port = _edge(main, output_port)
                parent = self.main.get(parent_name)
                if not isinstance(parent, dict) or parent.get("type") != "GraphOp":
                    raise ValueError("producer path crosses non-GraphOp main arithmetic or Cast")
                parent_main, parent_data, parent_netout, parent_enums, parent_nodes = self._interface(
                    parent_name, path_port=("output_desc", parent_port))
                for point, key in ((parent_main, "output_desc"), (parent_netout, "input_desc"),
                                   (parent_netout, "output_desc")):
                    self._require_path_descriptor(parent_name, point, key, parent_port, parent_enums, wanted,
                                                  reason="upstream GraphOp/NetOutput changes FP32 dtype, shape or layout")
                path.extend((
                    {"kind": "MAIN_GRAPHOP_INPUT", "owner": owner, "node": main["name"],
                     "port": output_port, "producer": parent_name, "producer_port": parent_port,
                     "descriptor": wanted},
                    {"kind": "MAIN_GRAPHOP_OUTPUT", "owner": parent_name, "node": parent_name,
                     "port": parent_port, "descriptor": wanted},
                    {"kind": "CPU_NETOUTPUT_INPUT", "owner": parent_name, "node": parent_netout["name"],
                     "port": parent_port, "descriptor": wanted}))
                producer, output_port = _edge(parent_netout, parent_port)
                owner = parent_name
            raise ValueError("CPU producer path exceeds partition bound")
        except (ValueError, KeyError, TypeError, IndexError) as error:
            result = {"status": "UNRESOLVED", "reason": str(error), "path": path}
            if isinstance(error, PartitionInterfaceError):
                result["details"] = error.details
            return result


def summarize_precision_failures(precision):
    """Group repeated interface failures without hiding any affected node.

    The report remains failed. Interface issues use the exact underlying
    mismatch as the key, not the caller's path port, so one malformed interface
    does not create hundreds of copies of the same diagnostic paragraph.
    """
    grouped, error_count = {}, 0
    failures = [row for row in precision.get("nodes", []) if row.get("status") != "MATCHED"]
    for row in failures:
        errors = row.get("errors", [])
        if not isinstance(errors, list):
            errors = [errors]
        for original in errors:
            error = (original if isinstance(original, dict) else
                     {"kind": "UNSTRUCTURED_ERROR", "message": str(original)})
            error_count += 1
            resolution = error.get("partition_resolution", {})
            if not isinstance(resolution, dict):
                resolution = {}
            details = resolution.get("details")
            if isinstance(details, dict):
                core = {key: value for key, value in details.items() if key != "relation_to_path"}
                signature = {"kind": error.get("kind"), "reason": resolution.get("reason"), "details": core}
            else:
                signature = {"kind": error.get("kind"), "reason": resolution.get("reason"),
                             "node": row.get("name"), "error": error}
            key = json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if key not in grouped:
                grouped[key] = {**signature, "occurrences": 0, "affected_nodes": [],
                                "affected_input_ports": [], "relations_to_path": []}
            group = grouped[key]
            group["occurrences"] += 1
            if row.get("name") not in group["affected_nodes"]:
                group["affected_nodes"].append(row.get("name"))
            occurrence = {"node": row.get("name"), "port": error.get("port")}
            if occurrence not in group["affected_input_ports"]:
                group["affected_input_ports"].append(occurrence)
            relation = details.get("relation_to_path") if isinstance(details, dict) else None
            if relation is not None and relation not in group["relations_to_path"]:
                group["relations_to_path"].append(relation)
    return {"status": precision.get("status"), "failed_node_count": len(failures),
            "error_count": error_count, "unique_issue_count": len(grouped),
            "issues": list(grouped.values())}
