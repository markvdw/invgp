import os
import re
from glob import glob


def get_next_filename(path, base_filename="data"):
    if not os.path.exists(path):
        os.makedirs(path)
    largest_existing_number = max([int(re.findall(r'\d+', fn)[-1]) for fn in glob(f"{path}/{base_filename}*")] + [0])
    return f"{path}/{base_filename}{largest_existing_number + 1}.json"
