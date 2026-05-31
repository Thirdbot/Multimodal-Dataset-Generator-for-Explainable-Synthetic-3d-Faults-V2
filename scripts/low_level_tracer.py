# tracing attribute connection happening in db named low-level because every change get logged
import json
import sqlite3

from yaml_helper import YAMLHelper
from pathlib import Path

class LowLevelTracer(object):
    def __init__(self,path):
        self.path = Path(path)
        self.parameter_db_path = Path(path) / "parameters.db"
        self.conn = sqlite3.connect(self.parameter_db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        self.sample_name = self.path.stem

        setting_path = Path(__file__).parent.parent.joinpath("settings.yaml")
        yaml_helper = YAMLHelper(setting_path)
        traces_path = yaml_helper.get_data('traces_path')
        self.traces_path = Path(traces_path)
        Path(traces_path).mkdir(parents=True,exist_ok=True)

        self.properties_data = self.get_data_in_table(self._get_table())

    def save_to_json(self):
        self.trace_sample_path = Path(self.traces_path.joinpath(f"{self.sample_name}_db_extract.json"))
        print(f"extracting properties from sample: {self.sample_name}")
        self.trace_sample_path.write_text(json.dumps(self.properties_data, indent=2, default=self._json_default))

    def get_properties_data(self):
        return self.properties_data

    def get_data_in_table(self,table):
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
        table = [
            row["name"] for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ] # get all table
        return table

# will be watcher later when update happens in output folder ,which will follow sequencing update laters if model change in chains
if __name__ == '__main__':
    root = Path(__file__).parent.parent.joinpath("outputs")
    candidates =  sorted(root.glob("seismic__*/"))
    first = candidates[1]

    low_tracker = LowLevelTracer(first)
    low_tracker.save_to_json()
