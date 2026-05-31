import yaml
import pandas as pd

class YAMLHelper:
    def __init__(self,yaml_path):
        self.yaml_path = yaml_path
        # take root from setting path (root-level)
        self.root = self.yaml_path.parent
        self.non_path_keys = ['control',
                              'category_counts',
                              'category_order',
                              'category_ratio',
                              'population_amount'] # non paths


    def _read_yaml(self):
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def get_key(self):
        data = self._read_yaml()
        return data.keys()

    def get_value(self):
        data = self._read_yaml()
        return data.values()

    def get_data(self,target):
        data = self._read_yaml()
        # level-1 sub handle
        df = pd.json_normalize(data).to_dict(orient='records')[0]
        prefix = f"{target}."
        matches = {k.replace(prefix, "") for k,v in df.items() if k.startswith(prefix)} # 1 level deep
        # list handler
        if target in data.keys():
            return data[target]

        if not matches:
            # intersect across blacklist
            non_path_keys = list(set(self.non_path_keys) & set(data.keys()))
            # 2 level deep
            found = next((v for k,v in df.items() if k == target or k.endswith(f".{target}")),None)
            # value that can't format to path
            for non_path_key in non_path_keys:
                if target in data[non_path_key]:
                    return found
            # can format to path
            return (self.root / found).as_posix()
        # return whole dict
        return matches

# test
if __name__ == "__main__":
    from pathlib import Path
    path = Path(__file__).parent.parent.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(path)
    print(yaml_helper.get_data('recipes_path'))