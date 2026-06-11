"""Dump every table from a Synthoseis parameters.db into JSON.

This is the low-level extraction layer. It does not interpret the DB content;
it preserves rows by table so graph_generator.py can decide what to keep.
"""

import json
import sqlite3

from yaml_helper import YAMLHelper
from pathlib import Path

class ParameterDbTracer(object):
    """Read one build folder's parameters.db and expose/save its tables."""

    def __init__(self,path):
        self.path = Path(path)
        self.parameter_db_path = Path(path) / "parameters.db"
        self.conn = sqlite3.connect(self.parameter_db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        self.sample_name = self.path.stem

        setting_path = Path(__file__).parent.parent.joinpath("settings.yaml")
        yaml_helper = YAMLHelper(setting_path)
        graphs_path = yaml_helper.get_data('graphs_path')
        self.graphs_path = Path(graphs_path)
        Path(graphs_path).mkdir(parents=True,exist_ok=True)

        self.properties_data = self._get_data_in_table(self._get_table())

    def save_to_json(self):
        """Write the extracted DB table payload to graphs_path as JSON."""
        self.db_extract_path = Path(self.graphs_path.joinpath(f"{self.sample_name}_db_extract.json"))
        print(f"extracting properties from sample: {self.sample_name}")
        self.db_extract_path.write_text(json.dumps(self.properties_data, indent=2, default=self._json_default))

    def _get_properties_data(self):
        return self.properties_data

    def _get_data_in_table(self,table):
        """Fetch all rows for each SQLite table name provided."""
        data = {}
        for table_name in table:
            row = self.conn.execute(
                f"SELECT * FROM {table_name}"
            ).fetchall()
            data[table_name] = [dict(row) for row in row]
        self.conn.close()
        return data

    @staticmethod
    def _json_default(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _get_table(self):
        """List all SQLite tables found in the parameters database."""
        table = [
            row["name"] for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ] # get all table
        return table


# Backward-compatible alias for older imports while scripts move to clearer names.
LowLevelTracer = ParameterDbTracer
