import argparse
import json
import os
import re
from pathlib import Path

from yaml_helper import YAMLHelper


ROOT = Path(__file__).parent.parent


def safe_label(value):
    text = re.sub(r"[^0-9A-Za-z_]", "_", str(value or "Unknown"))
    if not text:
        text = "Unknown"
    if text[0].isdigit():
        text = f"L_{text}"
    return text


def safe_rel_type(value):
    text = re.sub(r"[^0-9A-Za-z_]", "_", str(value or "RELATED_TO"))
    if not text:
        text = "RELATED_TO"
    if text[0].isdigit():
        text = f"R_{text}"
    return text.upper()


def latest_graph(graph_root):
    graph_paths = sorted(graph_root.glob("*_properties_graph.json"))
    if not graph_paths:
        raise FileNotFoundError(f"no properties graphs found under {graph_root}")
    return graph_paths[-1]


def load_graph_payload(graph_path):
    payload = json.loads(graph_path.read_text())
    payload.setdefault("nodes", [])
    payload.setdefault("edges", [])
    return payload


def clear_database(driver, database):
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")


def create_constraint(driver, database):
    query = """
    CREATE CONSTRAINT graph_node_id_unique IF NOT EXISTS
    FOR (n:GraphNode)
    REQUIRE n.id IS UNIQUE
    """
    with driver.session(database=database) as session:
        session.run(query)


def merge_node(driver, database, node):
    node_id = node["id"]
    label = safe_label(node.get("label", "Unknown"))
    props = {key: value for key, value in node.items() if key not in {"id", "label"}}
    props["source_label"] = node.get("label", "Unknown")

    query = f"""
    MERGE (n:GraphNode:{label} {{id: $id}})
    SET n += $props
    """
    with driver.session(database=database) as session:
        session.run(query, id=node_id, props=props)


def merge_edge(driver, database, edge):
    rel_type = safe_rel_type(edge.get("type", "RELATED_TO"))
    source = edge["source"]
    target = edge["target"]
    props = {key: value for key, value in edge.items() if key not in {"source", "target", "type"}}
    props["source_type"] = edge.get("type", "RELATED_TO")

    query = f"""
    MATCH (a:GraphNode {{id: $source}})
    MATCH (b:GraphNode {{id: $target}})
    MERGE (a)-[r:{rel_type}]->(b)
    SET r += $props
    """
    with driver.session(database=database) as session:
        session.run(query, source=source, target=target, props=props)


def import_graph(graph_path, uri, user, password, database, clear=False):
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise SystemExit(
            "neo4j driver is not installed. Install it with: uv add neo4j"
        ) from exc

    payload = load_graph_payload(graph_path)
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        driver.verify_connectivity()
        if clear:
            clear_database(driver, database)
        create_constraint(driver, database)

        for node in payload["nodes"]:
            merge_node(driver, database, node)

        for edge in payload["edges"]:
            merge_edge(driver, database, edge)
    finally:
        driver.close()

    return {
        "graph_path": graph_path.as_posix(),
        "node_count": len(payload["nodes"]),
        "edge_count": len(payload["edges"]),
        "database": database,
        "uri": uri,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a properties graph JSON into Neo4j for browser visualization."
    )
    parser.add_argument(
        "graph_path",
        nargs="?",
        help="Path to a *_properties_graph.json file. Defaults to the latest graph.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing Neo4j data before importing this graph.",
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j bolt URI. Default: %(default)s",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username. Default: %(default)s",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD", "neo4jneo4j"),
        help="Neo4j password. Default: env NEO4J_PASSWORD or %(default)s",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name. Default: %(default)s",
    )
    return parser.parse_args()


def resolve_graph_path(graph_path_arg):
    if graph_path_arg:
        return Path(graph_path_arg).expanduser().resolve()

    yaml_helper = YAMLHelper(ROOT / "settings.yaml")
    graph_root = Path(yaml_helper.get_data("traces_path")) / "properties_graph"
    return latest_graph(graph_root)


def print_browser_queries():
    print()
    print("Neo4j Browser queries:")
    print("MATCH (n:GraphNode) RETURN n LIMIT 50;")
    print("MATCH p=(s:Sample)-[*1..4]-(n) RETURN p LIMIT 100;")
    print("MATCH p=(f:FaultSystem)-[:HAS_FAULT]->(fault:Fault) RETURN p LIMIT 50;")
    print("MATCH p=(c:ClosureSystem)-[:HAS_CLOSURE]->(closure:Closure)-[:HAS_FLUID]->(fluid:Fluid) RETURN p LIMIT 100;")


def main():
    args = parse_args()
    graph_path = resolve_graph_path(args.graph_path)
    result = import_graph(
        graph_path=graph_path,
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        clear=args.clear,
    )

    print(json.dumps(result, indent=2))
    print_browser_queries()


if __name__ == "__main__":
    main()
