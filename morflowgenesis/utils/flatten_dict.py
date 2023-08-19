from copy import copy


def flatten_dict(input_dict, parent_key="", sep=".", special_chars=r".\|/;:+{}"):
    flattened_dict = {}
    for key, value in input_dict.items():
        if sep in special_chars:
            key = f"'{key}'"
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_dict(copy(value), new_key, sep))
        else:
            flattened_dict[new_key] = value
    return flattened_dict
