import cudf
import cupy as cp
from pygcylon import DataFrame, CylonEnv
from pygcylon.net.mpi_config import MPIConfig

def local_merge():
    df1 = cudf.DataFrame({'first': cp.random.rand(6), 'second': cp.random.rand(6)})
    df2 = cudf.DataFrame({'first': cp.random.rand(6), 'second': cp.random.rand(6)})
    print("df1: \n", df1)
    print("df2: \n", df2)
    cdf1 = DataFrame(df1)
    cdf2 = DataFrame(df2)
    cdf3 = cdf1.merge(right=cdf2, how="left", on=None, left_index=False, right_index=False)
    print("locally merged df: \n", cdf3.df)


def dist_merge():
    env: CylonEnv = CylonEnv(config=MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = cudf.DataFrame({'first': cp.random.rand(6), 'second': cp.random.rand(6)})
    df2 = cudf.DataFrame({'first': cp.random.rand(6), 'second': cp.random.rand(6)})

    arrays = [['a', 'a', 'b', 'b', 'c', 'c'], [1, 2, 3, 4, 5, 6]]
    tuples = list(zip(*arrays))
    idx = cudf.MultiIndex.from_tuples(tuples)
    # df1.index = idx
    # df2.index = idx
    print(df1)
    print(df2)
    cdf1 = DataFrame(df1)
    cdf2 = DataFrame(df2)
    cdf3 = cdf1.merge(right=cdf2, on="first", how="left", left_on=None, right_on=None, left_index=False, right_index=False, env=env)
    print("distributed joined df:\n", cdf3.df)
    env.finalize()

#####################################################
# local join test
local_merge()

# distributed join
#dist_merge()

# test_mpi()