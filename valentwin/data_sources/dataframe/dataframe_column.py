# Ported from Valentine: https://github.com/delftdata/valentine
# Added column type and expand_list

from valentwin.utils.utils import ColumnTypes, get_column_type
from ..base_column import BaseColumn


class DataframeColumn(BaseColumn):

    def __init__(self, column_name: str, data: list, d_type: str, table_guid: str, expand_list: bool = True):
        self.__column_name = column_name
        if expand_list and type(data[0]) == list:
            data = [item for sublist in data for item in sublist if item is not None]
        self.__data = data
        self.__d_type = d_type
        self.__table_guid = table_guid
        self.__c_type = get_column_type(data)

    @property
    def unique_identifier(self) -> str:
        return f"{self.__table_guid[0]}_{self.__table_guid[1]}:{self.__column_name}"

    @property
    def name(self):
        return self.__column_name

    @property
    def data_type(self):
        return self.__d_type

    @property
    def column_type(self) -> ColumnTypes:
        return self.__c_type

    @property
    def data(self) -> list:
        return self.__data
