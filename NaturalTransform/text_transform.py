import re


# ---------------------------------------------------------------------------
# Manual edit zone
# ---------------------------------------------------------------------------
# Keep this file as a small string builder. Add/remove evidence behavior here
# first, before touching the class logic below.

OBJECT_TYPES = ("fault", "closure")

MODEL_KEYS = {
    "number_faults",
    "fault_mode",
    "salt_inserted",
    "number_onlap_episodes",
    "number_fan_episodes",
    "number_hc_closures",
    "number_fault_intersections",
}

CLOSURE_KEYS = {
    "fluid",
    "intersects_fault",
    "intersects_onlap",
    "intersects_salt",
}

FAULT_KEYS = {
    "throw",
    "tilt_pct",
    "shear_zone_width",
    "gouge_pctile",
}

VISUAL_KEYS = set()

POSITION_EDGES = {"x", "y"}
EXTENT_EDGES = {"x_min", "x_max", "y_min", "y_max"}
SKIP_EDGES = {"view", "original_fault_index"}
LOW_VALUE_EXCEPTIONS = {"salt_inserted"}
ALLOWED_PROPERTY_EDGES = MODEL_KEYS | CLOSURE_KEYS | FAULT_KEYS | VISUAL_KEYS

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
    "fluid": "fluid",
    "salt_inserted": "salt",
    "number_faults": "faults",
    "number_hc_closures": "hydrocarbon closures",
    "number_onlap_episodes": "onlap episodes",
    "number_fault_intersections": "fault intersections",
    "number_fan_episodes": "fan episodes",
}

PROPERTY_TEMPLATES = {
    "fault_mode": "{source} is {value}",
    "throw": "{source} has throw of about {value}",
    "tilt_pct": "{source} shows tilt of about {value}",
    "shear_zone_width": "{source} has a shear zone about {value} wide",
    "gouge_pctile": "{source} shows gouge near the {value} percentile",
    "fluid": "{source} contains {value}",
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

EDGE_TEMPLATES = {
    "HAS_VISUAL_OBJECT": "{source} includes a visible {target} feature",
}

SPECIAL_TOKENS = {
    "object": ("<object>", "</object>"),
    "bbox": ("<bbox>", "</bbox>"),
    "center": ("<center>", "</center>"),
    "nums": ("<nums>", "</nums>"),
}


class TextTransform(object):
    """Convert graph relations into inspectable natural evidence strings."""

    def relations_to_evidence(self, relations):
        relations = list(relations)
        grouped_evidence, grouped_relation_ids = self._grouped_evidence(relations)
        evidence = list(grouped_evidence)

        for relation in relations:
            if id(relation) in grouped_relation_ids:
                continue

            sentence = self.relation_to_sentence(relation)
            if sentence:
                evidence.append(self._evidence_item(relation, sentence))

        return evidence

    def relation_to_sentence(self, relation):
        if relation.get("trace_type") == "edge":
            return self._edge_sentence(relation)

        edge = relation.get("edge")
        target = relation.get("target")
        if not self._include_property(edge, target):
            return None

        source = self.node_name(relation.get("source"))
        return self._property_sentence(source, edge, target)

    # Name and label helpers -------------------------------------------------

    def node_name(self, node_id):
        node_id = str(node_id)
        if node_id.startswith("category:"):
            return "the section"
        if node_id in NODE_NAMES:
            return self._tag("object", NODE_NAMES[node_id])

        match = re.match(r"^([a-z_]+)_(\d+)$", node_id)
        if match and match.group(1) in NUMBERED_NODE_NAMES:
            number = int(match.group(2)) + 1
            name = NUMBERED_NODE_NAMES[match.group(1)].format(number=number)
            return self._tag("object", name)

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

    # Sentence builders ------------------------------------------------------

    def _property_sentence(self, source, edge, target):
        if edge in BOOLEAN_TEMPLATES:
            return self._boolean_sentence(edge, target)
        if edge in COUNT_TEMPLATES:
            return self._count_sentence(edge, target)
        if edge in PROPERTY_TEMPLATES:
            return self._template_sentence(
                PROPERTY_TEMPLATES[edge],
                source,
                self._tag_number(target),
            )
        if str(edge).startswith("intersects_"):
            return self._intersection_sentence(source, edge, target)
        return None

    def _boolean_sentence(self, edge, target):
        if not self._is_true_value(target):
            return None
        return self._sentence(BOOLEAN_TEMPLATES[edge])

    def _count_sentence(self, edge, target):
        if self._is_false_value(target):
            return None

        noun = self.edge_label(edge)
        sentence = COUNT_TEMPLATES[edge].format(
            count=self._tag_number(target),
            noun=self._plural(noun, target),
        )
        return self._sentence(sentence)

    def _template_sentence(self, template, source, value):
        return self._sentence(template.format(
            source=source,
            value=value,
        ))

    def _intersection_sentence(self, source, edge, target):
        target_name = self.edge_label(edge).replace("intersects ", "")
        verb = "avoids" if self._is_false_value(target) else "intersects"
        return self._sentence(f"{source} {verb} {target_name}")

    def _edge_sentence(self, relation):
        edge = relation.get("edge")
        template = EDGE_TEMPLATES.get(edge)
        if not template:
            return None

        sentence = template.format(
            source=self.node_name(relation.get("source")),
            target=self.node_name(relation.get("target")),
        )
        return self._sentence(sentence)

    # Grouped evidence builders --------------------------------------------

    def _grouped_evidence(self, relations):
        evidence = []
        grouped_relation_ids = set()

        for item in self._position_evidence(relations):
            evidence.append(item)
            grouped_relation_ids.update(id(relation) for relation in item.pop("_group_relations"))

        for item in self._extent_evidence(relations):
            evidence.append(item)
            grouped_relation_ids.update(id(relation) for relation in item.pop("_group_relations"))

        return evidence, grouped_relation_ids

    def _position_evidence(self, relations):
        output = []
        for source_id, group in self._groups_for(relations, POSITION_EDGES).items():
            if not POSITION_EDGES.issubset(group):
                continue

            target = {edge: group[edge].get("target") for edge in POSITION_EDGES}
            sentence = (
                f"{self.node_name(source_id)} sits near "
                f"{self._center_text(target['x'], target['y'])}"
            )
            output.append(self._group_item(source_id, "position", group, self._sentence(sentence)))
        return output

    def _extent_evidence(self, relations):
        output = []
        for source_id, group in self._groups_for(relations, EXTENT_EDGES).items():
            if not EXTENT_EDGES.issubset(group):
                continue

            target = {edge: group[edge].get("target") for edge in EXTENT_EDGES}
            sentence = (
                f"{self.node_name(source_id)} occupies the area from "
                f"{self._bbox_text(target['x_min'], target['y_min'], target['x_max'], target['y_max'])}"
            )
            output.append(self._group_item(source_id, "extent", group, self._sentence(sentence)))
        return output

    @staticmethod
    def _groups_for(relations, edges):
        groups = {}
        for relation in relations:
            edge = relation.get("edge")
            if edge in edges:
                groups.setdefault(relation.get("source"), {})[edge] = relation
        return groups

    # Evidence object builders ---------------------------------------------

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

    # Include/filter helpers ------------------------------------------------

    def _include_property(self, edge, target):
        if edge not in ALLOWED_PROPERTY_EDGES and edge not in POSITION_EDGES and edge not in EXTENT_EDGES:
            return False
        if edge in SKIP_EDGES:
            return False
        return not self._is_low_value(edge, target)

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

    # Value formatting helpers ---------------------------------------------

    @classmethod
    def _bbox_text(cls, x_min, y_min, x_max, y_max):
        return cls._tag("bbox", [
            cls._value_text(x_min),
            cls._value_text(y_min),
            cls._value_text(x_max),
            cls._value_text(y_max),
        ])

    @classmethod
    def _center_text(cls, x, y):
        return cls._tag("center", [
            cls._value_text(x),
            cls._value_text(y),
        ])

    @classmethod
    def _tag_number(cls, value):
        return cls._tag("nums", cls._value_text(value)) if cls._is_number(value) else cls._value_text(value)

    @staticmethod
    def _tag(token_name, value):
        open_tag, close_tag = SPECIAL_TOKENS[token_name]
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        return f"{open_tag}{value}{close_tag}"

    @staticmethod
    def _is_number(value):
        if isinstance(value, bool):
            return False
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True

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
    def _is_true_value(value):
        return str(value).strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _is_false_value(value):
        return str(value).strip().lower() in {"0", "false", "no"}
