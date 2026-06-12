import re

import inflect
import simplenlg as nlg
from simplenlg.features import Feature, Tense
from simplenlg.lexicon.Lexicon import Lexicon


class TextTransform(object):
    SKIP_EDGES = {
        "a",
        "b",
        "c",
        "model_id",
        "cube_shape",
        "incident_angles",
        "sn_db",
        "bandpass_bandlimit_low",
        "bandpass_bandlimit_high",
        "fault_voxel_count_list",
        "closure_voxel_count",
        "closure_voxel_count_brine",
        "closure_voxel_count_oil",
        "closure_voxel_count_gas",
        "n_voxels",
        "n_voxels_faults",
        "n_voxels_fault_intersections",
        "salt_noise_stretch_factor",
        "intercept_avg",
        "gradient_avg",
        "view",
        "fixed_axis",
        "fixed_index",
        "selection_method",
        "image_path",
        "mask_path",
        "overlay_path",
        "overlay_image_path",
        "mask_image_path",
        "overlay_kind",
        "overlay_array",
        "overlay_arrays",
        "overlay_components",
        "wrapper_source",
        "original_fault_index",
    }

    PHRASES = {
        "fault_mode": ("the fault pattern", "be", "{value}"),
        "throw": ("{source}", "have", "throw of about {value}"),
        "tilt_pct": ("{source}", "show", "tilt of about {value}"),
        "shear_zone_width": ("{source}", "have", "a shear zone about {value} wide"),
        "gouge_pctile": ("{source}", "show", "gouge near the {value} percentile"),
        "sand_voxel_pct": ("sand-prone interval", "make up", "about {value} percent of the section"),
        "sand_layer_percent_a_posteriori": ("sand-prone layering", "make up", "about {value} of the section"),
        "closure_voxel_pct": ("closure expression", "cover", "about {value} percent of the section"),
        "closure_voxel_count": ("closure expression", "cover", "about {value} voxels"),
        "n_voxels": ("{source}", "occupy", "about {value} voxels"),
        "n_voxels_faults": ("fault zone", "cover", "about {value} voxels"),
        "fluid": ("{source}", "contain", "{value}"),
    }

    def __init__(self):
        self.inflect = inflect.engine()
        self.lexicon = Lexicon.getDefaultLexicon()
        self.factory = nlg.NLGFactory(self.lexicon)
        self.realiser = nlg.Realiser(self.lexicon)

    def relations_to_sentences(self, relations):
        sentences = []
        skip = self.grouped_relation_ids(relations)
        sentences.extend(self.grouped_position_sentences(relations))

        for relation in relations:
            if id(relation) in skip:
                continue
            sentence = self.relation_to_sentence(relation)
            if sentence:
                sentences.append(sentence)
        return sentences

    def grouped_relation_ids(self, relations):
        skip = set()
        for group in self.position_groups(relations).values():
            skip.update(id(relation) for relation in group.values())
        for group in self.extent_groups(relations).values():
            skip.update(id(relation) for relation in group.values())
        return skip

    def grouped_position_sentences(self, relations):
        sentences = []

        for source_id, group in self.position_groups(relations).items():
            sentence = self.position_sentence(source_id, group)
            if sentence:
                sentences.append(sentence)

        for source_id, group in self.extent_groups(relations).items():
            sentence = self.extent_sentence(source_id, group)
            if sentence:
                sentences.append(sentence)

        return sentences

    def position_sentence(self, source_id, group):
        source = self.node_name(source_id)
        if all(key in group for key in ("x0", "y0", "z0")):
            x = self.number_text(group["x0"].get("target"))
            y = self.number_text(group["y0"].get("target"))
            z = self.number_text(group["z0"].get("target"))
            return self.realise(source, "sit", f"near x={x}, y={y}, and z={z}")
        if all(key in group for key in ("x", "y")):
            x = self.number_text(group["x"].get("target"))
            y = self.number_text(group["y"].get("target"))
            return self.realise(source, "sit", f"near x={x} and y={y}")
        return None

    def extent_sentence(self, source_id, group):
        source = self.node_name(source_id)
        required_3d = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
        if required_3d.issubset(group):
            x0 = self.number_text(group["x_min"].get("target"))
            x1 = self.number_text(group["x_max"].get("target"))
            y0 = self.number_text(group["y_min"].get("target"))
            y1 = self.number_text(group["y_max"].get("target"))
            z0 = self.number_text(group["z_min"].get("target"))
            z1 = self.number_text(group["z_max"].get("target"))
            return self.realise(source, "span", f"x={x0} to {x1}, y={y0} to {y1}, and z={z0} to {z1}")
        required_2d = {"x_min", "x_max", "y_min", "y_max"}
        if required_2d.issubset(group):
            x0 = self.number_text(group["x_min"].get("target"))
            x1 = self.number_text(group["x_max"].get("target"))
            y0 = self.number_text(group["y_min"].get("target"))
            y1 = self.number_text(group["y_max"].get("target"))
            return self.realise(source, "occupy", f"the area from x={x0} to {x1} and y={y0} to {y1}")
        return None

    def position_groups(self, relations):
        groups = {}
        for relation in relations:
            edge = relation.get("edge")
            if edge not in {"x0", "y0", "z0", "x", "y"}:
                continue
            groups.setdefault(relation.get("source"), {})[edge] = relation
        return groups

    def extent_groups(self, relations):
        groups = {}
        for relation in relations:
            edge = relation.get("edge")
            if edge not in {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}:
                continue
            groups.setdefault(relation.get("source"), {})[edge] = relation
        return groups

    def relations_to_evidence(self, relations):
        evidence = []
        skip = self.grouped_relation_ids(relations)
        evidence.extend(self.grouped_position_evidence(relations))
        for relation in relations:
            if id(relation) in skip:
                continue
            sentence = self.relation_to_sentence(relation)
            if not sentence:
                continue
            evidence.append({
                **relation,
                "source": relation.get("trace_type", ""),
                "fact_name": relation.get("edge", ""),
                "value": relation.get("target", ""),
                "sentence": sentence,
            })
        return evidence

    def grouped_position_evidence(self, relations):
        evidence = []
        for source_id, group in self.position_groups(relations).items():
            sentence = self.position_sentence(source_id, group)
            if sentence:
                evidence.append(self.group_evidence(source_id, "position", group, sentence))
        for source_id, group in self.extent_groups(relations).items():
            sentence = self.extent_sentence(source_id, group)
            if sentence:
                evidence.append(self.group_evidence(source_id, "extent", group, sentence))
        return evidence

    def group_evidence(self, source_id, fact_name, group, sentence):
        return {
            "trace_type": "property_group",
            "source": source_id,
            "edge": fact_name,
            "target": {
                key: relation.get("target")
                for key, relation in group.items()
            },
            "relation": [
                source_id,
                fact_name,
                {
                    key: relation.get("target")
                    for key, relation in group.items()
                },
            ],
            "fact_name": fact_name,
            "value": {
                key: relation.get("target")
                for key, relation in group.items()
            },
            "sentence": sentence,
        }

    def relation_to_sentence(self, relation):
        edge = relation.get("edge")
        if edge in self.SKIP_EDGES:
            return None
        if self.is_low_value(edge, relation.get("target")):
            return None

        source = self.node_name(relation.get("source"))
        target = relation.get("target")

        if relation.get("trace_type") == "edge":
            return self.edge_sentence(source, edge, self.node_name(target))

        return self.property_sentence(source, edge, target)

    def property_sentence(self, source, edge, target):
        label = self.edge_label(edge)
        value = self.value_text(target)

        phrase = self.phrase_sentence(source, edge, target, value)
        if phrase:
            return phrase

        if self.is_count_edge(edge):
            phrase = self.count_phrase(target, self.count_noun(edge))
            return self.realise(source, "show", phrase)

        if self.is_position_edge(edge):
            return self.realise(source, "pass", f"near {self.axis(edge)}={value}")

        if self.is_extent_edge(edge):
            return self.realise(source, "reach", f"the {self.extent_side(edge)} {self.axis(edge)} side near {value}")

        if self.is_intersection_edge(edge):
            target_name = label.replace("intersects ", "")
            if self.is_false_value(target):
                return self.realise(source, "avoid", target_name)
            return self.realise(source, "intersect", target_name)

        if self.is_amount_edge(edge):
            return self.realise(source, "have", f"{label} of about {value}")

        if self.is_false_value(target):
            return self.realise(source, "show", f"no {self.boolean_noun(edge)}")

        if self.is_true_value(target):
            return self.realise(source, "show", self.boolean_noun(edge))

        return self.realise(source, "show", f"{label} {value}")

    def edge_sentence(self, source, edge, target):
        label = self.edge_label(edge)
        if edge.startswith("HAS_"):
            return None
        if edge == "REALIZED":
            return None
        return self.realise(source, label, target)

    def phrase_sentence(self, source, edge, target, value):
        if edge == "salt_inserted":
            if self.is_true_value(target):
                return self.realise("salt", "be", "present")
            return None

        if edge == "number_faults":
            if self.is_false_value(target):
                return None
            return self.realise("the section", "show", self.count_phrase(target, "fault"))

        if edge == "number_hc_closures":
            if self.is_false_value(target):
                return None
            return self.realise("the section", "contain", self.count_phrase(target, "hydrocarbon closure"))

        if edge == "number_onlap_episodes":
            if self.is_false_value(target):
                return None
            return self.realise("the layering", "show", self.count_phrase(target, "onlap episode"))

        if edge == "number_fault_intersections":
            if self.is_false_value(target):
                return None
            return self.realise("faults", "intersect", self.count_phrase(target, "place"))

        template = self.PHRASES.get(edge)
        if not template:
            return None

        subject, verb, object_ = template
        subject = subject.format(source=source, value=value)
        object_ = object_.format(source=source, value=value)
        return self.realise(subject, verb, object_)

    def realise(self, subject, verb, object_):
        clause = self.factory.createClause()
        clause.setSubject(subject)
        clause.setVerb(verb)
        clause.setObject(object_)
        clause.setFeature(Feature.TENSE, Tense.PRESENT)
        return self.finish(self.realiser.realise(clause).getRealisation())

    def node_name(self, node_id):
        node_id = str(node_id)
        if node_id.startswith("category:"):
            return "the section"
        if node_id == "fault":
            return "the fault zone"
        if node_id.startswith("fault_"):
            return f"fault {int(node_id.rsplit('_', 1)[1]) + 1}"
        if node_id == "closure":
            return "the closure zone"
        if node_id.startswith("closure_"):
            return f"closure {int(node_id.rsplit('_', 1)[1]) + 1}"
        return node_id.replace("_", " ")

    def edge_label(self, edge):
        edge = str(edge)
        if edge.startswith("HAS_"):
            return edge[4:].lower().replace("_", " ")
        if edge.endswith("_inserted"):
            edge = edge.removesuffix("_inserted")
        if edge.startswith("number_"):
            edge = edge.removeprefix("number_")
        words = edge.lower().split("_")
        replacements = {
            "hc": "hydrocarbon",
            "pct": "percentage",
            "pctile": "percentile",
            "avg": "average",
            "n": "number of",
            "voxels": "voxel",
            "mode": "pattern",
        }
        return " ".join(replacements.get(word, word) for word in words)

    def value_text(self, value):
        if isinstance(value, bool):
            return str(value).lower()
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value).replace("_", "-")

        if number.is_integer() and abs(number) < 10000:
            return self.inflect.number_to_words(int(number))
        return str(round(number, 4))

    @staticmethod
    def number_text(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value).replace("_", "-")
        if number.is_integer():
            return str(int(number))
        return str(round(number, 4))

    def count_phrase(self, value, noun):
        try:
            count = int(float(value))
        except (TypeError, ValueError):
            return f"{value} {self.inflect.plural(noun)}"
        number = self.inflect.number_to_words(count)
        noun = self.singular(noun)
        noun = self.inflect.plural_noun(noun, count) or noun
        return f"{number} {noun}"

    @staticmethod
    def is_count_edge(edge):
        edge = str(edge)
        return edge.startswith("number_") or edge.endswith("_count")

    def count_noun(self, edge):
        edge = str(edge)
        edge = re.sub(r"^number_", "", edge)
        edge = re.sub(r"_count$", "", edge)
        return self.edge_label(edge)

    def boolean_noun(self, edge):
        edge = str(edge)
        if edge.endswith("_inserted"):
            edge = edge.removesuffix("_inserted")
        return self.edge_label(edge)

    def singular(self, noun):
        words = str(noun).split()
        if not words:
            return noun
        singular = self.inflect.singular_noun(words[-1]) or words[-1]
        return " ".join([*words[:-1], singular])

    @staticmethod
    def is_position_edge(edge):
        return str(edge) in {"x0", "y0", "z0", "x", "y"}

    @staticmethod
    def is_extent_edge(edge):
        return str(edge) in {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}

    @staticmethod
    def is_intersection_edge(edge):
        return str(edge).startswith("intersects_")

    @staticmethod
    def is_amount_edge(edge):
        edge = str(edge)
        return (
            edge.endswith("_pct")
            or edge.endswith("_pctile")
            or edge.endswith("_width")
            or edge.endswith("_avg")
            or edge in {"throw", "n_voxels"}
            or edge.startswith("n_voxels_")
        )

    @staticmethod
    def is_low_value(edge, value):
        edge = str(edge)
        if edge in {"salt_inserted"}:
            return False
        if edge.startswith("intersects_"):
            return False
        if edge.startswith("number_") and edge not in {"number_faults"}:
            return str(value).strip().lower() in {"0", "0.0"}
        if edge.endswith("_pct") or edge.startswith("n_voxels_") or edge.endswith("_count"):
            return str(value).strip().lower() in {"0", "0.0"}
        return False

    @staticmethod
    def axis(edge):
        return str(edge)[0]

    @staticmethod
    def extent_side(edge):
        return "low" if str(edge).endswith("_min") else "high"

    @staticmethod
    def is_true_value(value):
        return str(value).strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def is_false_value(value):
        return str(value).strip().lower() in {"0", "false", "no"}

    @staticmethod
    def finish(sentence):
        sentence = str(sentence).strip()
        if not sentence:
            return sentence
        return sentence[0].upper() + sentence[1:]
