import os
import io
import re

import numpy as np
import pandas as pd

from sklearn.utils import Bunch
from sklearn.preprocessing import LabelEncoder

DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'keel')

def parse_keel_dat(dat_file):
    with open(dat_file, "r") as fp:
        data = fp.read()
        header, payload = data.split("@data\n")

    attributes = re.findall(
        r"@[Aa]ttribute (.*?)[ {](integer|real|.*)", header)
    output = re.findall(r"@[Oo]utput[s]? (.*)", header)

    dtype_map = {"integer": np.int, "real": np.float}

    columns, types = zip(*attributes)
    types = [*map(lambda _: dtype_map.get(_, np.object), types)]
    dtype = dict(zip(columns, types))

    data = pd.read_csv(io.StringIO(payload), names=columns, dtype=dtype)

    if not output:  # if it was not found
        output = columns[-1]

    target = data[output]
    data.drop(labels=output, axis=1, inplace=True)

    return data, target


def prepare_X_y(data, target):
    class_encoder = LabelEncoder()
    target = class_encoder.fit_transform(target.values.ravel())
    return data.values, target


def find_datasets(storage=DATASETS_DIR):
    for f_name in os.listdir(storage):
        yield f_name.split('.')[0]


def load_dataset(ds_name, return_X_y=False, storage=DATASETS_DIR):
    data_file = os.path.join(storage, f"{ds_name}.dat")
    data, target = parse_keel_dat(data_file)

    if return_X_y:
        return prepare_X_y(data, target)

    return Bunch(data=data, target=target, filename=data_file)

if __name__ == '__main__':
    import pandas as pd
    table = []

    for ds in find_datasets():
        data = load_dataset(ds)
        _, counts = np.unique(data.target, return_counts=True)
        ir  = np.divide(*counts)

        table.append({
            "Dataset": ds,
            "Majority": counts[0],
            "Minority": counts[1],
            "IR": ir,
        })

    df = pd.DataFrame(table).set_index('Dataset').sort_values('IR')
    df['IR'] = df["IR"].map("{:.2f}".format)
    print(df.to_markdown())
