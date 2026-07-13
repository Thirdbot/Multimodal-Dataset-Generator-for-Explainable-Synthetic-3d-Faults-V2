import re

# ---------------------------------------------------------------------------
# Small string builder: one sentence per graph relation (triplet). Keep it
# minimal -- no grammar variation (singular/plural). Numeric unit conversion
# (tilt %, throw / infill_factor) happens upstream in graph_system.
# ---------------------------------------------------------------------------

OBJECT_TYPES = ("fault", "closure", "salt")

MODEL_KEYS = {"number_faults", "fault_mode", "salt_inserted",
              "number_hc_closures", "number_fault_intersections"}
CLOSURE_KEYS = {"fluid", "intersects_fault", "intersects_onlap", "intersects_salt",
                "area_pct"}  # area_pct read from the closure mask (not in the DB)
# throw from the DB; dip_deg from the fault mask (replaces the opaque tilt_pct fraction).
FAULT_KEYS = {"throw", "dip_deg"}  # shear_zone_width / gouge_pctile dropped: sub-seismic, not visible
ALLOWED_PROPERTY_EDGES = MODEL_KEYS | CLOSURE_KEYS | FAULT_KEYS

POSITION_EDGES = {"x", "y"}
EXTENT_EDGES = {"x_min", "x_max", "y_min", "y_max"}
SKIP_EDGES = {"view", "original_fault_index"}
LOW_VALUE_EXCEPTIONS = {"salt_inserted"}

# Synthoseis fault_mode values that name a real geological pattern (Parameters.py:768).
# "random" is a generation setting ("as random as it can be"), not a structure, so it
# gets no pattern sentence -- asserting it would leak the simulator into the evidence.
FAULT_PATTERN_NAMES = {
    "relay_ramp": "relay ramp",
    "horst_and_graben": "horst-and-graben",
    "self_branching": "branching",
    "stair_case": "staircase",
}

# Descriptive dip-magnitude scale (USGS/ODP): gentle 0-30, moderate 30-60, steep 60-90.
# NOT the high/low-angle *fault* split (45 deg) -- and dip_deg is APPARENT dip measured
# in a time section, so readings are scoped "in this section", never a genetic claim.
DIP_STEEP_MIN = 60
DIP_GENTLE_MAX = 30

NODE_NAMES = {"fault": "fault", "closure": "closure", "salt": "salt"}
NUMBERED_NODE_NAMES = {"fault": "Fault {number}", "closure": "Closure {number}", "salt": "Salt {number}"}

EDGE_LABELS = {
    "throw": "throw",
    "dip_deg": "dip",
    "area_pct": "size",
    "fluid": "fluid",
    "salt_inserted": "salt",
    "number_faults": "faults",
    "number_hc_closures": "hydrocarbon closures",
    "number_fault_intersections": "fault intersections",
}

PROPERTY_TEMPLATES = {
    "throw": "{source} has throw of about {value} ms",  # ms two-way time (throw/infill x digi); axis is TWT
    "dip_deg": "{source} dips at about {value} degrees",
    "area_pct": "{source} covers about {value} percent of the section",
    "fluid": "{source} contains {value}",
}
COUNT_TEMPLATES = {
    "number_faults": "The section shows {count} {noun}",
    "number_hc_closures": "The section contains {count} {noun}",
    "number_fault_intersections": "Faults intersect {count} {noun}",
}
BOOLEAN_TEMPLATES = {"salt_inserted": "Salt is present"}
EDGE_TEMPLATES = {"HAS_VISUAL_OBJECT": "{source} includes a visible {target} feature"}

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
        grouped, grouped_ids = self._grouped_evidence(relations)
        evidence = list(grouped)
        for relation in relations:
            if id(relation) in grouped_ids:
                continue
            sentence = self.relation_to_sentence(relation)
            if sentence:
                evidence.append(self._evidence_item(relation, sentence))
                # Tier-2: deterministic geological readings derived from this verified
                # fact. Emitted only when the base fact was rendered (so they are always
                # grounded), and they carry the SAME object_id, so they mask the same
                # object and are retrievable + NLI-checkable like any evidence line.
                for reading in self._reading_sentences(relation):
                    evidence.append(self._reading_item(relation, reading))
        return evidence

    def _reading_sentences(self, relation):
        # Source-backed, definitional readings only (see DIP_* / trap notes). Never a
        # genetic or process claim -- those belong in the (un-checked) reasoning tier.
        edge = relation.get("edge")
        target = relation.get("target")
        source = self.node_name(relation.get("source"))
        readings = []
        if edge == "dip_deg":
            try:
                dip = float(target)
            except (TypeError, ValueError):
                dip = None
            if dip is not None:
                if dip >= DIP_STEEP_MIN:
                    term = "steeply dipping"
                elif dip < DIP_GENTLE_MAX:
                    term = "gently dipping"
                else:
                    term = "moderately dipping"
                readings.append(f"{source} appears {term} in this section")
        elif edge == "fluid":
            fluid = str(target).strip().lower()
            if fluid in {"oil", "gas"}:
                readings.append(f"{source} is a hydrocarbon-bearing closure")
            elif fluid == "brine":
                readings.append(f"{source} is a water-bearing closure")
        elif edge == "intersects_fault" and self._is_true(target):
            # Synthoseis uses intersects_fault to set closure TYPE (Closures.py:1562),
            # so "fault-dependent" is grounded in the simulator's own trap logic.
            readings.append(f"{source} is a fault-dependent closure")
        elif edge == "intersects_onlap" and self._is_true(target):
            readings.append(f"{source} is an onlap trap")
        return readings

    def _reading_item(self, relation, sentence):
        return {
            "trace_type": "reading",
            "source": relation.get("source", ""),
            "object_id": self._object_id(relation),
            "edge": "reading",
            "target": relation.get("target"),
            "relation": [relation.get("source"), "reading", relation.get("target")],
            "fact_name": "reading",
            "value": relation.get("target"),
            "sentence": sentence,
        }

    def relation_to_sentence(self, relation):
        if relation.get("trace_type") == "edge":
            return self._edge_sentence(relation)
        edge, target = relation.get("edge"), relation.get("target")
        if not self._include_property(edge, target):
            return None
        return self._property_sentence(self.node_name(relation.get("source")), edge, target)

    # Names ------------------------------------------------------------------

    def node_name(self, node_id):
        node_id = str(node_id)
        if node_id.startswith("category:"):
            return "the section"
        if node_id in NODE_NAMES:
            return self._tag("object", NODE_NAMES[node_id])
        match = re.match(r"^([a-z_]+)_(\d+)$", node_id)
        if match and match.group(1) in NUMBERED_NODE_NAMES:
            name = NUMBERED_NODE_NAMES[match.group(1)].format(number=int(match.group(2)) + 1)
            return self._tag("object", name)
        return node_id.replace("_", " ")

    def edge_label(self, edge):
        edge = str(edge)
        if edge in EDGE_LABELS:
            return EDGE_LABELS[edge]
        return edge.removeprefix("number_").removesuffix("_inserted").replace("_", " ")

    # Sentence builders ------------------------------------------------------

    def _property_sentence(self, source, edge, target):
        if edge == "fault_mode":
            return self._fault_mode_sentence(source, target)
        if edge in BOOLEAN_TEMPLATES:
            return self._sentence(BOOLEAN_TEMPLATES[edge]) if self._is_true(target) else None
        if edge in COUNT_TEMPLATES:
            if self._is_false(target):
                return None
            return self._sentence(COUNT_TEMPLATES[edge].format(
                count=self._tag_number(target), noun=self.edge_label(edge)))
        if edge in PROPERTY_TEMPLATES:
            return self._sentence(PROPERTY_TEMPLATES[edge].format(
                source=source, value=self._tag_number(target)))
        if str(edge).startswith("intersects_"):
            name = self.edge_label(edge).replace("intersects ", "")
            verb = "avoids" if self._is_false(target) else "intersects"
            return self._sentence(f"{source} {verb} {name}")
        return None

    def _fault_mode_sentence(self, source, target):
        # fault_mode is categorical; "none" is a real fact (no faulting) -> absence.
        # Only genuine geological patterns get a "pattern" sentence; "random"/unknown
        # are generator settings, not structures, so they are dropped (fall back to the
        # plain fault count elsewhere).
        value = str(target).strip().lower()
        if value in {"none", "", "0", "false"}:
            return self._sentence(f"{source} shows no faulting")
        name = FAULT_PATTERN_NAMES.get(value)
        if not name:
            return None
        return self._sentence(f"{source} shows a {name} fault pattern")

    def _edge_sentence(self, relation):
        template = EDGE_TEMPLATES.get(relation.get("edge"))
        if not template:
            return None
        return self._sentence(template.format(
            source=self.node_name(relation.get("source")),
            target=self.node_name(relation.get("target")),
        ))

    # Grouped position/extent -----------------------------------------------

    def _grouped_evidence(self, relations):
        evidence, grouped_ids = [], set()
        for edges, builder in ((POSITION_EDGES, self._position_sentence),
                               (EXTENT_EDGES, self._extent_sentence)):
            for source_id, group in self._groups_for(relations, edges).items():
                if not edges.issubset(group):
                    continue
                target = {edge: group[edge].get("target") for edge in edges}
                evidence.append(self._group_item(source_id, edges, group, builder(source_id, target)))
                grouped_ids.update(id(relation) for relation in group.values())
        return evidence, grouped_ids

    def _position_sentence(self, source_id, target):
        center = self._tag("center", [self._value_text(target["x"]), self._value_text(target["y"])])
        return self._sentence(f"{self.node_name(source_id)} sits near {center}")

    def _extent_sentence(self, source_id, target):
        box = self._tag("bbox", [self._value_text(target["x_min"]), self._value_text(target["y_min"]),
                                 self._value_text(target["x_max"]), self._value_text(target["y_max"])])
        return self._sentence(f"{self.node_name(source_id)} occupies the area from {box}")

    @staticmethod
    def _groups_for(relations, edges):
        groups = {}
        for relation in relations:
            if relation.get("edge") in edges:
                groups.setdefault(relation.get("source"), {})[relation.get("edge")] = relation
        return groups

    # Evidence item builders -------------------------------------------------

    def _evidence_item(self, relation, sentence):
        return {
            **relation,
            "trace_type": relation.get("trace_type", ""),
            "source": relation.get("source", ""),
            "object_id": self._object_id(relation),
            "fact_name": relation.get("edge", ""),
            "value": relation.get("target", ""),
            "sentence": sentence,
        }

    def _group_item(self, source_id, edges, group, sentence):
        fact_name = "position" if edges == POSITION_EDGES else "extent"
        target = {edge: relation.get("target") for edge, relation in group.items()}
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
        }

    @staticmethod
    def _object_id(relation):
        source, target = str(relation.get("source", "")), str(relation.get("target", ""))
        pattern = rf"^({'|'.join(OBJECT_TYPES)})_\d+$"
        if re.match(pattern, source) or source in OBJECT_TYPES:
            return source
        if re.match(pattern, target):
            return target
        return source

    # Include / filter -------------------------------------------------------

    def _include_property(self, edge, target):
        if edge in SKIP_EDGES:
            return False
        if edge not in ALLOWED_PROPERTY_EDGES and edge not in POSITION_EDGES and edge not in EXTENT_EDGES:
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
        if edge.endswith("_pct") or edge.endswith("_count") or edge.startswith("n_voxels_"):
            return value in {"0", "0.0"}
        return False

    # Value formatting -------------------------------------------------------

    @classmethod
    def _tag_number(cls, value):
        return cls._tag("nums", cls._value_text(value)) if cls._is_number(value) else cls._value_text(value)

    @staticmethod
    def _tag(token_name, value):
        open_tag, close_tag = SPECIAL_TOKENS[token_name]
        if isinstance(value, (list, tuple)):
            return f"{open_tag}[{','.join(str(item) for item in value)}]{close_tag}"
        return f"{open_tag}{value}{close_tag}"

    @staticmethod
    def _is_number(value):
        if isinstance(value, bool):
            return False
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _value_text(value):
        if isinstance(value, bool):
            return str(value).lower()
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value).replace("_", "-")
        return str(int(number)) if number.is_integer() else str(round(number, 4))

    @staticmethod
    def _sentence(text):
        text = str(text).strip()
        return text[:1].upper() + text[1:] if text else text

    @staticmethod
    def _is_true(value):
        return str(value).strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _is_false(value):
        return str(value).strip().lower() in {"0", "false", "no"}
