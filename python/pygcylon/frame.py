
import cudf
from cudf import _lib as libcudf
from pygcylon.data.table import shuffle

class CDataFrame(object):
    def __init__(self, df):
        if df and isinstance(df, cudf.DataFrame):
            self._df = df
        else:
            raise ValueError('A cudf.DataFrame object must be provided.')

    def shuffle(self, hash_columns, context):
        tbl = shuffle(self._df, hash_columns, context)
        out = cudf.DataFrame()
        out = out._from_table(tbl)
        return out
