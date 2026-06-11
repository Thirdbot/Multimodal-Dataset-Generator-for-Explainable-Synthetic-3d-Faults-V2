"""Runtime guard around Synthoseis config/build behavior.

The wrapper normalizes category intent before calling third_party/synthoseis.
It also temporarily patches fault generation so no-fault and low-fault-count
categories behave according to the generated config.
"""

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).parent.parent
SYNTHOSEIS_ROOT = ROOT / "third_party" / "synthoseis"
if str(SYNTHOSEIS_ROOT) not in sys.path:
    sys.path.insert(0, str(SYNTHOSEIS_ROOT))


CATEGORY_RULES = {
    "boring": {
        "include_salt": False,
        "basin_floor_fans": False,
        "include_channels": False,
        "min_number_faults": 0,
        "max_number_faults": 0,
        "closure_types": ["simple"],
    },
    "salt_only": {
        "include_salt": True,
        "basin_floor_fans": False,
        "include_channels": False,
        "min_number_faults": 0,
        "max_number_faults": 0,
        "closure_types": ["simple"],
    },
    "onlap": {
        "include_salt": False,
        "basin_floor_fans": False,
        "include_channels": False,
        "min_number_faults": 0,
        "max_number_faults": 0,
        "closure_types": ["onlap"],
    },
    "depositional": {
        "include_salt": False,
        "basin_floor_fans": True,
        "include_channels": False,
        "min_number_faults": 0,
        "max_number_faults": 0,
        "closure_types": ["simple", "onlap"],
    },
}


def _category_from_name(path):
    """Infer a dataset category from a config filename."""
    stem = Path(path).stem
    for category in sorted(CATEGORY_RULES, key=len, reverse=True):
        if stem == category or stem.startswith(f"{category}_") or f"_{category}_" in stem:
            return category
    return None


def _normalized_config(config_path):
    """Load JSON config and apply category-level guard rules."""
    config_path = Path(config_path)
    with open(config_path, "r") as file:
        config = json.load(file)

    category = _category_from_name(config_path)
    if category in CATEGORY_RULES:
        config.update(CATEGORY_RULES[category])

    min_faults = int(config.get("min_number_faults") or 0)
    max_faults = int(config.get("max_number_faults") or 0)
    no_faults = max_faults <= 0

    # Synthoseis has internal fault modes that ignore small max values.
    # For low-count mixed examples, force the random mode so the JSON range wins.
    strict_fault_range = not no_faults and max_faults < 6

    return config, {
        "category": category,
        "no_faults": no_faults,
        "strict_fault_range": strict_fault_range,
        "min_faults": min_faults,
        "max_faults": max_faults,
    }


@contextmanager
def _fault_settings_override(build_rules):
    """Temporarily patch Synthoseis fault settings during one build call."""
    from datagenerator.Parameters import Parameters

    original_fault_settings = Parameters._fault_settings

    def _guarded_fault_settings(self):
        if build_rules["no_faults"]:
            self.low_fault_throw = 0.0
            self.high_fault_throw = 0.0
            self.mode = 0
            self.clustering = 0
            self.number_faults = 0
            self.fmode = "none"
            self.fault_param = ["00", 0, 0.0, 0.0]
            return

        if build_rules["strict_fault_range"]:
            self.low_fault_throw = 5.0 * self.infill_factor
            self.high_fault_throw = 35.0 * self.infill_factor
            self.mode = 0
            self.clustering = 0
            if build_rules["max_faults"] <= build_rules["min_faults"]:
                self.number_faults = build_rules["min_faults"]
            else:
                self.number_faults = self.rng.integers(
                    build_rules["min_faults"],
                    build_rules["max_faults"],
                )
            self.fmode = "random"
            self.fault_param = [
                "00",
                self.number_faults,
                self.low_fault_throw,
                self.high_fault_throw,
            ]
            return

        return original_fault_settings(self)

    Parameters._fault_settings = _guarded_fault_settings
    try:
        yield
    finally:
        Parameters._fault_settings = original_fault_settings


def guarded_build_model(build_model, user_json, run_id, test_mode=None, rpm_factors=None, seed=None):
    """Call Synthoseis build_model with a guarded temporary config file."""
    config, build_rules = _normalized_config(user_json)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"guarded_{Path(user_json).stem}_",
        dir="/tmp",
        delete=False,
    ) as file:
        json.dump(config, file, indent=2)
        guarded_json = file.name

    try:
        with _fault_settings_override(build_rules):
            return build_model(
                user_json=guarded_json,
                run_id=run_id,
                test_mode=test_mode,
                rpm_factors=rpm_factors,
                seed=seed,
            )
    finally:
        Path(guarded_json).unlink(missing_ok=True)
