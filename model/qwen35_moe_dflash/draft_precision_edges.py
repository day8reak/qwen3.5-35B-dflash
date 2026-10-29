"""Prove FP32 producer identity across explicitly owned CPUCL partitions.

This supports the observed packed Data/NetOutput interface: one Data operation
per CPU GraphDef exposes the ordered GraphOp inputs, and one NetOutput exposes
the ordered outputs. The complete interface descriptors must agree. No node-name
guessing, shape-only producer matching, Cast traversal or opaque NPU traversal is
accepted. Binary task ownership is established by the caller before construction.
"""
from __future__ import annotations

import re


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

    def _interface(self, owner):
        if owner not in self.cpu:
            raise ValueError("upstream partition has no explicit CPUCL GraphDef ownership")
        item, nodes = self.cpu[owner]
        main = self.main.get(owner)
        if not isinstance(main, dict) or main.get("type") != "GraphOp":
            raise ValueError("CPU GraphDef has no matching main GraphOp owner")
        attrs = main.get("attr", [])
        if not isinstance(attrs, list):
            raise ValueError("main GraphOp ownership attributes are invalid")
        libraries = []
        for row in attrs:
            if not isinstance(row, dict) or row.get("key") != "cl_name":
                continue
            value = row.get("value")
            if not isinstance(value, dict):
                raise ValueError("main GraphOp CPUCL ownership attribute is malformed")
            libraries.append(value.get("s"))
        if libraries != ["CPUCL"]:
            raise ValueError("main GraphOp does not have unique CPUCL ownership")
        data = [node for node in nodes.values() if node.get("type") == "Data"]
        netout = [node for node in nodes.values() if node.get("type") == "NetOutput"]
        if len(data) != 1 or len(netout) != 1:
            raise ValueError("CPU packed interface requires one Data and one NetOutput")
        enums = item["descriptor_enum_fields"]
        for child, main_key in ((data[0], "input_desc"), (netout[0], "output_desc")):
            parent_descs = main.get(main_key)
            if not isinstance(parent_descs, list) or len(parent_descs) > 8192:
                raise ValueError("main GraphOp interface descriptors are missing or exceed bound")
            count = len(parent_descs)
            if any(not isinstance(child.get(key), list) or len(child[key]) != count
                   for key in ("input_desc", "output_desc")):
                raise ValueError("CPU packed interface port counts differ from its GraphOp")
            for port in range(count):
                parent = self._descriptor(main, main_key, port, enums)
                if any(self._descriptor(child, key, port, enums) != parent
                       for key in ("input_desc", "output_desc")):
                    raise ValueError("CPU packed interface descriptors differ from its GraphOp")
        return main, data[0], netout[0], enums, nodes

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
                main, data, netout, enums, nodes = self._interface(owner)
                node = nodes.get(producer)
                if not isinstance(node, dict):
                    raise ValueError("CPU partition producer is unresolved")
                descriptor = self._descriptor(node, "output_desc", output_port, enums)
                if descriptor != wanted:
                    raise ValueError("producer path changes FP32 dtype or shape")
                path.append({"kind": "CPU_OUTPUT", "owner": owner, "node": producer,
                             "port": output_port, "descriptor": descriptor})
                if producer == expected["producer"] and output_port == expected_port:
                    if node.get("type") in {"Data", "NetOutput", "GraphOp", "Cast", "CastT"}:
                        raise ValueError("declared arithmetic producer resolves to an interface or Cast")
                    return {"status": "MATCHED", "owner": owner, "producer": producer,
                            "output_port": output_port, "path": path}
                if node is not data:
                    raise ValueError("cross-partition path reaches an undeclared operation or Cast")
                if self._descriptor(data, "input_desc", output_port, enums) != wanted:
                    raise ValueError("CPU Data input changes FP32 dtype or shape")
                if self._descriptor(main, "input_desc", output_port, enums) != wanted:
                    raise ValueError("main GraphOp input changes FP32 dtype or shape")
                parent_name, parent_port = _edge(main, output_port)
                parent = self.main.get(parent_name)
                if not isinstance(parent, dict) or parent.get("type") != "GraphOp":
                    raise ValueError("producer path crosses non-GraphOp main arithmetic or Cast")
                parent_main, parent_data, parent_netout, parent_enums, parent_nodes = self._interface(parent_name)
                for point, key in ((parent_main, "output_desc"), (parent_netout, "input_desc"),
                                   (parent_netout, "output_desc")):
                    if self._descriptor(point, key, parent_port, parent_enums) != wanted:
                        raise ValueError("upstream GraphOp/NetOutput changes FP32 dtype or shape")
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
            return {"status": "UNRESOLVED", "reason": str(error), "path": path}
