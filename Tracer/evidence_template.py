# minimal template for graph evidence during tracing

def round_value(value, digit=4):
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return round(value, digit)
    return value


def sample_template(node):
    sample = node.get("sample_id", node["id"])
    return f"Sample {sample} is a generated seismic model sample."


def property_template(node):
    label = node.get("label")
    if label == "Category":
        return f"The sample category is {node.get('name', 'unknown')}."
    if label == "ModelParameters":
        attrs = []
        if node.get("cube_shape"):
            attrs.append(f"cube shape {node.get('cube_shape')}")
        if node.get("number_faults") is not None:
            attrs.append(f"{node.get('number_faults')} configured faults")
        if node.get("fault_mode"):
            attrs.append(f"fault mode {node.get('fault_mode')}")
        if node.get("salt_inserted") is not None:
            attrs.append(f"salt inserted={node.get('salt_inserted')}")
        if node.get("number_onlap_episodes") is not None:
            attrs.append(f"{node.get('number_onlap_episodes')} onlap episodes")
        if node.get("number_hc_closures") is not None:
            attrs.append(f"{node.get('number_hc_closures')} hydrocarbon closures")
        text = ", ".join(attrs) if attrs else "recorded model settings"
        return f"Model parameters include {text}."
    if label == "FaultSystem":
        count = node.get("realized_faults", "unknown")
        return f"The sample realizes a fault system with {count} faults."
    if label == "Fault":
        fault_index = node.get("fault_index", "unknown")
        x0 = round_value(node.get("x0"))
        y0 = round_value(node.get("y0"))
        z0 = round_value(node.get("z0"))
        throw = round_value(node.get("throw"))
        return (
            f"Fault {fault_index} is centered near x={x0}, y={y0}, z={z0} "
            f"with throw {throw}."
        )
    if label == "ClosureSystem":
        count = node.get("realized_closures", "unknown")
        return f"The sample realizes a closure system with {count} closures."
    if label == "Closure":
        closure_index = node.get("closure_index", "unknown")
        fluid = node.get("fluid", "unknown")
        n_voxels = node.get("n_voxels", "unknown")
        x_min = round_value(node.get("x_min"))
        x_max = round_value(node.get("x_max"))
        y_min = round_value(node.get("y_min"))
        y_max = round_value(node.get("y_max"))
        z_min = round_value(node.get("z_min"))
        z_max = round_value(node.get("z_max"))
        return (
            f"Closure {closure_index} contains {fluid} and spans "
            f"x={x_min} to {x_max}, y={y_min} to {y_max}, z={z_min} to {z_max}, "
            f"with {n_voxels} voxels."
        )
    if label == "Fluid":
        return f"The closure fluid type is {node.get('name', 'unknown')}."

    attrs = []
    for key, value in node.items():
        if key in {"id", "label", "source_graphs"}:
            continue
        if value in (None, "", [], {}):
            continue
        attrs.append(f"{key}={round_value(value)}")
        if len(attrs) >= 8:
            break
    text = ", ".join(attrs) if attrs else "no recorded attributes"
    return f"{label} has {text}."

def relation_template(edge, source_node, target_node):
    edge_type = edge.get("type")
    source_label = source_node.get("label")
    target_label = target_node.get("label")

    if edge_type == "HAS_CATEGORY":
        return f"The sample belongs to the {target_node.get('name', 'unknown')} category."
    if edge_type == "REALIZED" and target_label == "FaultSystem":
        return f"The build realizes a fault system with {target_node.get('realized_faults', 'unknown')} faults."
    if edge_type == "REALIZED" and target_label == "ClosureSystem":
        return f"The build realizes a closure system with {target_node.get('realized_closures', 'unknown')} closures."
    if edge_type == "HAS_FAULT":
        return (
            f"The fault system includes fault {target_node.get('fault_index', 'unknown')} "
            f"centered near x={round_value(target_node.get('x0'))}, "
            f"y={round_value(target_node.get('y0'))}, z={round_value(target_node.get('z0'))}."
        )
    if edge_type == "HAS_CLOSURE":
        return (
            f"The closure system includes closure {target_node.get('closure_index', 'unknown')} "
            f"with fluid {target_node.get('fluid', 'unknown')}."
        )
    if edge_type == "HAS_FLUID":
        return (
            f"Closure {source_node.get('closure_index', 'unknown')} contains "
            f"{target_node.get('name', 'unknown')}."
        )

    return f"{source_label} has relation {edge_type} to {target_label}."
