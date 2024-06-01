# Ported from Valentine: https://github.com/delftdata/valentine
# Added convert_to_final_df, read_matches_df, annotate_tp_fp, ColumnTypes, get_column_type, convert_to_number,
# convert_str_to_list

from ast import literal_eval
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import List

import pandas as pd

def is_sorted(matches: dict):
    prev = None
    for value in matches.values():
        if prev is None:
            prev = value
        else:
            if prev > value:
                return False
    return True


def convert_data_type(string: str):
    try:
        f = float(string)
        if f.is_integer():
            return int(f)
        return f
    except ValueError:
        return string.replace("\n", "\\n")


def normalize_distance(dist: int,
                       str1: str,
                       str2: str):
    """
    Function that returns a normalized similarity score between two strings given their distance

    Parameters
    ----------
    dist : int
        The distance between the two strings (hamming, levenshtein or damerau levenshtein)
    str1: str
        The first string
    str2: str
        The second string
    """

    return 1 - dist/max(max(len(str1), len(str2)), 1)


def get_project_root():
    return str(Path(__file__).parent.parent)


def convert_to_final_df(matches, tables, table_names, with_values=False):
    rows = []
    for ((table_a, column_a), (table_b, column_b), column_type), sim in matches.items():
        if with_values:
            row = {
                "table_a": table_a,
                "column_a": column_a,
                "values_a": list(tables[table_names.index(table_a)][column_a].dropna())[:10],
                "table_b": table_b,
                "column_b": column_b,
                "values_b": list(tables[table_names.index(table_b)][column_b].dropna())[:10],
                "similarity": sim
            }
        else:
            row = {
                "table_a": table_a,
                "column_a": column_a,
                "table_b": table_b,
                "column_b": column_b,
                "similarity": sim
            }
        if column_type is not None:
            row["type"] = column_type
        rows.append(row)
    matches_df = pd.DataFrame(rows)
    return matches_df


def read_matches_df(filepath):
    matches_df = pd.read_csv(filepath, index_col=0)
    matches = {}
    for i, row in matches_df.iterrows():
        matches[((row["table_a"], row["column_a"]), (row["table_b"], row["column_b"]))] = row["similarity"]
    return matches


def annotate_tp_fp(matches_df, ground_truth_pairs, correct_col_name="correct"):
    matches_df[correct_col_name] = False
    ground_truth_pairs = deepcopy(ground_truth_pairs)
    ground_truth_pairs += [(pair[1], pair[0]) for pair in ground_truth_pairs]
    for i, row in matches_df.iterrows():
        if ((row["table_a"], row["column_a"]), (row["table_b"], row["column_b"])) in ground_truth_pairs:
            matches_df.loc[i, correct_col_name] = True
    return matches_df


class ColumnTypes(Enum):
    TEXTUAL = "textual"
    NUMERICAL = "numerical"
    MIXED = "mixed"


COLUMN_VALUE_PLACEHOLDERS = ['tbc', 'none', 'n/a', 'unknown', '-']

def get_column_type(attribute, column_threshold=0.5, entity_threshold=0.5):
    """
    This function takes a column and determines whether it is text or numeric column
    This has been done using a well-known information retrieval technique
    Check each cell to see if it is text. Then if enough number of cells are
    text, the column is considered as a text column.
    *Taken from ALITE*
    """
    attribute = pd.Series(attribute).dropna().tolist() # drop null values

    # if all items are in placeholders, then it is a textual column
    if all(item in COLUMN_VALUE_PLACEHOLDERS for item in attribute):
        return ColumnTypes.TEXTUAL

    # other normal string values
    filtered_attribute = [item for item in attribute if
                          item is not None and (type(item) != str or item.lower() not in COLUMN_VALUE_PLACEHOLDERS)]
    str_attribute = [item for item in attribute if type(item) == str and item.lower() not in COLUMN_VALUE_PLACEHOLDERS]
    str_att = [item for item in str_attribute if not item.isdigit() and len(item) > 0]
    for i in range(len(str_att)-1, -1, -1):
        entity = str_att[i]
        num_count = 0
        for char in entity:
            if char.isdigit():
                num_count += 1
        if num_count/len(entity) > entity_threshold:
            del str_att[i]
    if len(str_att)/len(filtered_attribute) > column_threshold:  # if is of textual type
        return ColumnTypes.TEXTUAL

    # check for numerical values
    try:  # if is of numerical type
        converted = [convert_to_number(item) for item in filtered_attribute]
        return ColumnTypes.NUMERICAL
    except ValueError:
        return ColumnTypes.MIXED
    # if len(str_att) > 0:  # if is of mixed type
    #     return ColumnTypes.MIXED
    # else: # if is of numerical type
    #     return ColumnTypes.NUMERICAL


def convert_to_number(s):
    # Remove commas
    s_no_commas = s.replace(',', '') if type(s) == str else s
    # Convert to float
    num_float = float(s_no_commas)
    return num_float


def convert_str_to_list(df: pd.DataFrame, column_names: List[str] = None):
    if column_names is None:
        column_names = df.columns
    for col in column_names:
        if col in df.columns:
            sample_col = df[df[col].notnull()][col]
            if len(sample_col) > 0:
                sample = df[df[col].notnull()][col].iloc[0]
                if type(sample) == str and sample.startswith("[") and sample.endswith("]"):
                    try:
                        df[col] = df[col].apply(literal_eval)
                    except:
                        # df[col] = df[col].apply(
                        #     lambda x: x.strip("[]").replace("'", "").split(", ") if type(x) == str and x != '[]' else list())
                        df[col] = df[col].apply(
                            lambda x: [y[1:] for y in x.strip("[]")[:-1].replace('", ', "', ").split("', ")]
                            if type(x) == str and x != '[]' and x != '[None]' and x != 'None' else list())
                    # convert dict to str for every item in the list
                    df[col] = df[col].apply(lambda x: [str(y) if type(y) == dict else y for y in x])
                # convert dict to str
                df[col] = df[col].apply(lambda x: str(x) if type(x) == dict else x)
    return df


