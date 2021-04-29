import cudf
import cupy as cp
from pygcylon import DataFrame, CylonEnv
from pygcylon.net.mpi_config import MPIConfig

def local_concat():
    df1 = cudf.DataFrame({'first': cp.random.rand(6), 'second': cp.random.rand(6)})
    df2 = cudf.DataFrame({'first': cp.random.rand(6), 'second': cp.random.rand(6)})
    print("df1: \n", df1)
    print("df2: \n", df2)
    cdf1 = DataFrame(df1)
    cdf2 = DataFrame(df2)
    cdf3 = cdf1.merge(right=cdf2, how="left", on=None, left_index=False, right_index=False)
    print("locally merged df: \n", cdf3.df)


def dist_concat():
    env: CylonEnv = CylonEnv(config=MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = cudf.DataFrame({'first': cp.random.rand(6), 'second': cp.random.rand(6)})
    df2 = cudf.DataFrame({'second': cp.random.rand(6), 'first': cp.random.rand(6)})
    print(df1)
    print(df2)
    cdf1 = DataFrame(df1)
    cdf2 = DataFrame(df2)
    cdf3 = DataFrame.concat([cdf1, cdf2], join="inner", env=env)
    print("distributed concated df:\n", cdf3.df)
    env.finalize()

#####################################################
# local join test
# local_concat()

# distributed join
dist_concat()
