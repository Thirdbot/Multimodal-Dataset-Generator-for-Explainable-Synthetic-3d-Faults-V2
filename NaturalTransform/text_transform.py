import re


# Manual edit zone. Keep these maps small and inspectable.
OBJECT_TYPES = ("fault", "closure")

SKIP_EDGES = {
    "fault_voxel_count_list",
    "closure_voxel_count",
    "closure_voxel_count_brine",
    "closure_voxel_count_oil",
    "closure_voxel_count_gas",
    "closure_voxel_pct",
    "closure_voxel_pct_brine",
    "closure_voxel_pct_oil",
    "closure_voxel_pct_gas",
    "n_voxels",
    "n_voxels_faults",
    "n_voxels_fault_intersections",
    "view",
    "original_fault_index",
}

LOW_VALUE_EXCEPTIONS = {"salt_inserted"}
POSITION_EDGES = {"x", "y"}
EXTENT_EDGES = {"x_min", "x_max", "y_min", "y_max"}

NODE_NAMES = {
    "fault": "the fault zone",
    "closure": "the closure zone",
}

NUMBERED_NODE_NAMES = {
    "fault": "Fault {number}",
    "closure": "Closure {number}",
}

EDGE_LABELS = {
    "fault_mode": "fault pattern",
    "throw": "throw",
    "tilt_pct": "tilt",
    "shear_zone_width": "shear zone width",
    "gouge_pctile": "gouge percentile",
    "sand_voxel_pct": "sand-prone interval percentage",
    "sand_layer_percent_a_posteriori": "sand-prone layering percentage",
    "fluid": "fluid",
    "salt_inserted": "salt",
    "number_faults": "faults",
    "number_hc_closures": "hydrocarbon closures",
    "number_onlap_episodes": "onlap episodes",
    "number_fault_intersections": "fault intersections",
    "number_fan_episodes": "fan episodes",
    "onlaps_horizon_list": "onlap horizons",
    "fan_horizon_list": "fan horizons",
}

EDGE_TEMPLATES = {
    "fault_mode": "{source} is {value}",
    "throw": "{source} has throw of about {value}",
    "tilt_pct": "{source} shows tilt of about {value}",
    "shear_zone_width": "{source} has a shear zone about {value} wide",
    "gouge_pctile": "{source} shows gouge near the {value} percentile",
    "sand_voxel_pct": "Sand-prone intervals make up about {value} percent of the section",
    "sand_layer_percent_a_posteriori": "Sand-prone layering makes up about {value} of the section",
    "fluid": "{source} contains {value}",
    "onlaps_horizon_list": "Onlap is associated with horizons {value}",
    "fan_horizon_list": "Fan deposition is associated with horizons {value}",
}

COUNT_TEMPLATES = {
    "number_faults": "The section shows {count} {noun}",
    "number_hc_closures": "The section contains {count} {noun}",
    "number_onlap_episodes": "The layering shows {count} {noun}",
    "number_fault_intersections": "Faults intersect {count} {noun}",
    "number_fan_episodes": "The section shows {count} {noun}",
}

BOOLEAN_TEMPLATES = {
    "salt_inserted": "Salt is present",
}


class TextTransform(object):
    def relations_to_evidence(self, relations):
        relations = list(relations)
        evidence = self._grouped_evidence(relations)
        grouped_ids = {id(relation) for item in evidence for relation in item.pop("_group_relations")}

        for relation in relations:
            if id(relation) in grouped_ids:
                continue
            sentence = self.relation_to_sentence(relation)
            if sentence:
                evidence.append(self._evidence_item(relation, sentence))
        return evidence

    def relation_to_sentence(self, relation):
        edge = relation.get("edge")
        target = relation.get("target")
        if edge in SKIP_EDGES or self._is_low_value(edge, target):
            return None

        if relation.get("trace_type") == "edge":
            return None

        source = self.node_name(relation.get("source"))
        return self._property_sentence(source, edge, target)

    def node_name(self, node_id):
        node_id = str(node_id)
        if node_id.startswith("category:"):
            return "the section"
        if node_id in NODE_NAMES:
            return NODE_NAMES[node_id]

        match = re.match(r"^([a-z_]+)_(\d+)$", node_id)
        if match and match.group(1) in NUMBERED_NODE_NAMES:
            number = int(match.group(2)) + 1
            return NUMBERED_NODE_NAMES[match.group(1)].format(number=number)
        return node_id.replace("_", " ")

    def edge_label(self, edge):
        edge = str(edge)
        if edge in EDGE_LABELS:
            return EDGE_LABELS[edge]
        if edge.endswith("_inserted"):
            edge = edge.removesuffix("_inserted")
        if edge.startswith("number_"):
            edge = edge.removeprefix("number_")
        return edge.replace("_", " ")

    def _property_sentence(self, source, edge, target):
        value = self._value_text(target)

        if edge in BOOLEAN_TEMPLATES:
            return BOOLEAN_TEMPLATES[edge] if self._is_true_value(target) else None

        if edge in COUNT_TEMPLATES:
            if self._is_false_value(target):
                return None
            noun = self.edge_label(edge)
            return COUNT_TEMPLATES[edge].format(count=self._count_text(target), noun=self._plural(noun, target))

        if edge in EDGE_TEMPLATES:
            return self._sentence(EDGE_TEMPLATES[edge].format(source=source, value=value))

        if str(edge).startswith("intersects_"):
            target_name = self.edge_label(edge).replace("intersects ", "")
            verb = "avoids" if self._is_false_value(target) else "intersects"
            return self._sentence(f"{source} {verb} {target_name}")

        return None

    def _grouped_evidence(self, relations):
        evidence = []
        for source_id, group in self._groups_for(relations, POSITION_EDGES).items():
            if POSITION_EDGES.issubset(group):
                target = {edge: group[edge].get("target") for edge in POSITION_EDGES}
                sentence = (
                    f"{self.node_name(source_id)} sits near "
                    f"x={self._value_text(target['x'])} and y={self._value_text(target['y'])}"
                )
                evidence.append(self._group_item(source_id, "position", group, self._sentence(sentence)))

        for source_id, group in self._groups_for(relations, EXTENT_EDGES).items():
            if EXTENT_EDGES.issubset(group):
                target = {edge: group[edge].get("target") for edge in EXTENT_EDGES}
                sentence = (
                    f"{self.node_name(source_id)} occupies the area from "
                    f"x={self._value_text(target['x_min'])} to {self._value_text(target['x_max'])} "
                    f"and y={self._value_text(target['y_min'])} to {self._value_text(target['y_max'])}"
                )
                evidence.append(self._group_item(source_id, "extent", group, self._sentence(sentence)))
        return evidence

    @staticmethod
    def _groups_for(relations, edges):
        groups = {}
        for relation in relations:
            edge = relation.get("edge")
            if edge in edges:
                groups.setdefault(relation.get("source"), {})[edge] = relation
        return groups

    def _evidence_item(self, relation, sentence):
        return {
            **relation,
            "trace_type": relation.get("trace_type", ""),
            "source": relation.get("source", ""),
            "object_id": self._object_id_from_relation(relation),
            "fact_name": relation.get("edge", ""),
            "value": relation.get("target", ""),
            "sentence": sentence,
        }

    def _group_item(self, source_id, fact_name, group, sentence):
        target = {key: relation.get("target") for key, relation in group.items()}
        return {
            "trace_type": "property_group",
            "source": source_id,
            "object_id": source_id,
            "edge": fact_name,
            "target": target,
            "relation": [source_id, fact_name, target],
            "fact_name": fact_name,
            "value": target,
            "sentence": sentence,
            "_group_relations": list(group.values()),
        }

    @staticmethod
    def _object_id_from_relation(relation):
        source = str(relation.get("source", ""))
        target = str(relation.get("target", ""))
        object_pattern = rf"^({'|'.join(OBJECT_TYPES)})_\d+$"
        if re.match(object_pattern, source):
            return source
        if re.match(object_pattern, target):
            return target
        if source in OBJECT_TYPES:
            return source
        return source

    @staticmethod
    def _value_text(value):
        if isinstance(value, bool):
            return str(value).lower()
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value).replace("_", "-")
        if number.is_integer():
            return str(int(number))
        return str(round(number, 4))

    @staticmethod
    def _count_text(value):
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return str(value)
        return str(number)

    @staticmethod
    def _plural(noun, value):
        try:
            count = int(float(value))
        except (TypeError, ValueError):
            count = 2
        if count == 1:
            return noun[:-1] if noun.endswith("s") else noun
        return noun if noun.endswith("s") else f"{noun}s"

    @staticmethod
    def _sentence(text):
        text = str(text).strip()
        if not text:
            return text
        return text[0].upper() + text[1:]

    @staticmethod
    def _is_low_value(edge, value):
        edge = str(edge)
        if edge in LOW_VALUE_EXCEPTIONS or edge.startswith("intersects_"):
            return False
        value = str(value).strip().lower()
        if edge.startswith("number_") and edge != "number_faults":
            return value in {"0", "0.0"}
        if edge.endswith("_pct") or edge.startswith("n_voxels_") or edge.endswith("_count"):
            return value in {"0", "0.0"}
        return False

    @staticmethod
    def _is_true_value(value):
        return str(value).strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _is_false_value(value):
        return str(value).strip().lower() in {"0", "false", "no"}
