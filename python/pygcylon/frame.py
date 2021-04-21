
import cudf
from cudf import _lib as libcudf
from pygcylon.data.table import shuffle
from pygcylon.ctx.context import CylonContext

class CylonEnv(object):

    def __init__(self, config=None, distributed=True) -> None:
        self._context = CylonContext(config, distributed)
        self._distributed = distributed
        self._finalized = False

    @property
    def context(self) -> CylonContext:
        return self._context

    @property
    def rank(self) -> int:
        return self._context.get_rank()

    @property
    def world_size(self) -> int:
        return self._context.get_world_size()

    @property
    def is_distributed(self) -> bool:
        return self._distributed

    def finalize(self):
        if not self._finalized:
            self._finalized = True
            self._context.finalize()

    def barrier(self):
        self._context.barrier()

    def __del__(self):
        """
        On destruction of the application, the environment will be automatically finalized
        """
        self.finalize()


class DataFrame(object):

    def __init__(self, df):
        if df and isinstance(df, cudf.DataFrame):
            self._df = df
        else:
            raise ValueError('A cudf.DataFrame object must be provided.')

    @property
    def df(self) -> cudf.DataFrame:
        return self._df

    def shuffle(self, hash_columns, env: CylonEnv = None) -> cudf.DataFrame:
        tbl = shuffle(self._df, hash_columns, env.context)
        return cudf.DataFrame._from_table(tbl)

    def join(self, other, on=None, how='left', lsuffix='l', rsuffix='r', sort=False, algorithm="hash", env: CylonEnv = None):

        if env is None:
            joined_df = self._df.join(other=other._df, on=on, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort, method=algorithm)
            return DataFrame(joined_df)

        # shuffle dataframes on index columns
        hash_columns = [*range(self._df._num_indices)]
        shuffledDf1 = self.shuffle(hash_columns, env)

        hash_columns = [*range(other._df._num_indices)]
        shuffledDf2 = other.shuffle(hash_columns, env)

        joined_df = shuffledDf1.join(shuffledDf2, on=on, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort, method=algorithm)
        return DataFrame(joined_df)
