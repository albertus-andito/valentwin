from transformers import AutoTokenizer
from typing import Any, Dict, Tuple

import pandas as pd


class ColumnToTextTransformer:

    def __init__(self, all_tables: Dict[str, pd.DataFrame], tokenizer: AutoTokenizer, max_length: int = 512,
                 column_contexts: Dict[str, Dict[str, str]] = None):
        self.all_tables = all_tables
        self.frequency_dictionary = self._create_frequency_dictionary()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.column_contexts = column_contexts

    def _create_frequency_dictionary(self, method: str = "document_frequency") -> Dict[Any, int]:
        # document_frequency: number of columns among all tables that have the cell value
        if method == "document_frequency":
            value_columns_dict = {}
            for table_name, table in self.all_tables.items():
                for column in table.columns:
                    # if column values are lists, flatten them
                    if len(table[column]) > 0 and type(table[column].tolist()[0]) == list:
                        column_values = [item for sublist in table[column].tolist() for item in sublist if
                                         item is not None]
                        table[column] = pd.Series(column_values)
                    for value in table[column].unique():
                        # Ensure the value is hashable (convert if necessary)
                        value = hash(value) if not isinstance(value, (int, float, str, tuple)) else value
                        if value in value_columns_dict:
                            value_columns_dict[value].add(column)
                        else:
                            value_columns_dict[value] = {column}
            frequency_dict = {value: len(columns) for value, columns in value_columns_dict.items()}
            return frequency_dict
        else:
            raise ValueError(f"Method {method} not supported")

    def get_all_column_representations(self, method: str = "title-colname-stat-col",
                                       tables: Dict[str, pd.DataFrame] = None,
                                       shuffle_column_values: bool = False) -> Dict[str, Dict[str, str]]:
        column_representations = {}
        if tables is None:
            tables = self.all_tables
        for table_name, table in tables.items():
            table_column_representations = {}
            for column in table.columns:
                if shuffle_column_values:
                    table[column] = table[column].sample(frac=1).reset_index(drop=True)
                if method == "col":
                    table_column_representations[column] = self.transform_to_col(table[column])
                elif method == "colname-col":
                    table_column_representations[column] = self.transform_to_col(table[column], column)
                elif method == "colname-col-context":
                    table_column_representations[column] = self.transform_to_colname_col_context(table[column], col_name=column,
                                                                                           context=self.column_contexts.get(table_name, {}).get(column, ''))
                elif method == "colname-stat-col":
                    table_column_representations[column] = self.transform_to_colname_stat_col(table[column], col_name=column)
                elif method == "title-colname-col":
                    table_column_representations[column] = self.transform_to_title_colname_col(table[column], col_name=column,
                                                                                         table_title=table_name)
                elif method == "title-colname-col-context":
                    table_column_representations[column] = self.transform_to_title_colname_col_context(table[column], col_name=column,
                                                                                                 table_title=table_name,
                                                                                                 context=self.column_contexts.get(table_name, {}).get(column, ''))
                elif method == "title-colname-stat-col":
                    table_column_representations[column] = self.transform_to_title_colname_stat_col(table[column], col_name=column,
                                                                                             table_title=table_name)
                else:
                    raise ValueError(f"Method {method} not supported")
            column_representations[table_name] = table_column_representations
        return column_representations

    def transform_to_col(self, column: pd.Series, initial_transformed_col: str = '', suffix_to_transformed_col: str = '') -> str:
        column = column.dropna()
        if len(column) > 0 and type(column.tolist()[0]) == list:
            column = [item for sublist in column.tolist() for item in sublist if item is not None]
            column = pd.Series(column)
        transformed_col, last_value_index = self._concat_until_max_length(column, initial_transformed_col, suffix_to_transformed_col)
        if last_value_index < len(column) - 1:
            # If not all values are included, sort column values by frequency
            # This is in line with what the paper says: "In the case of a tall input column, we choose a frequency-based approach".
            column = column.sort_values(key=lambda x: x.map(lambda y: (-self.frequency_dictionary.get(y, 0), y)))
            transformed_col, last_value_index = self._concat_until_max_length(column, initial_transformed_col,
                                                                              suffix_to_transformed_col)
        return transformed_col

    def _concat_until_max_length(self, column: pd. Series, initial_transformed_col: str = '', suffix_to_transformed_col: str = '') -> Tuple[str, int]:
        transformed_col = initial_transformed_col
        last_index = -1
        for i, text in enumerate(column):
            # Check token count if this text is added
            new_text_without_suffix = transformed_col + ', ' + str(text) if i > 0 else transformed_col + ' ' + str(text)
            new_text = new_text_without_suffix + suffix_to_transformed_col
            token_count = len(self.tokenizer(new_text)["input_ids"])
            if token_count > self.max_length:
                break
            transformed_col = new_text_without_suffix
            last_index = i

        return transformed_col.strip(), last_index

    def transform_to_colname_col(self, column: pd.Series, col_name: str) -> str:
        transformed_col = col_name+ ": "
        return self.transform_to_col(column, transformed_col)

    def transform_to_colname_col_context(self, column: pd.Series, col_name: str, context: pd.DataFrame) -> str:
        # context denotes the accompanied context of the table (e.g., a brief description)
        transformed_col = col_name+ ": "
        suffix_text = ". " + context
        return self.transform_to_col(column, transformed_col, suffix_text) + suffix_text

    def transform_to_colname_stat_col(self, column: pd.Series, col_name: str) -> str:
        stats = self._calculate_column_statistics(column)
        transformed_col = f"{col_name} contains {stats[0]} values ({stats[1]}, {stats[2]}, {stats[3]:.2f}): "
        return self.transform_to_col(column, transformed_col)

    def transform_to_title_colname_col(self, column: pd.Series, col_name: str, table_title: str) -> str:
        transformed_col = f"{table_title}. {col_name}: "
        return self.transform_to_col(column, transformed_col)

    def transform_to_title_colname_col_context(self, column: pd.Series, col_name: str, table_title: str, context: str) -> str:
        transformed_col = f"{table_title}. {col_name}: "
        suffix_text = ". " + context
        return self.transform_to_col(column, transformed_col, suffix_text) + suffix_text

    def transform_to_title_colname_stat_col(self, column: pd.Series, col_name: str, table_title: str) -> str:
        stats = self._calculate_column_statistics(column)
        transformed_col = f"{table_title}. {col_name} contains {stats[0]} values ({stats[1]}, {stats[2]}, {stats[3]:.2f}): "
        return self.transform_to_col(column, transformed_col)

    def _calculate_column_statistics(self, column: pd.Series) -> Tuple[int, int, int, float]:
        # calculates num of values, as well as maximum, minimum, and average number of characters in a cell
        max_length = column.map(lambda x: len(str(x))).max()
        min_length = column.map(lambda x: len(str(x))).min()
        avg_length = column.map(lambda x: len(str(x))).mean()
        return len(column), max_length, min_length, avg_length
