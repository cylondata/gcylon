import cudf
import cupy as cp
from pygcylon import DataFrame, CylonEnv
from pygcylon.net.mpi_config import MPIConfig

def drop_cuplicates():
    env: CylonEnv = CylonEnv(config=MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)
    df1 = cudf.DataFrame({'first': cp.random.randint(100, 110, 20), 'second': cp.random.randint(100, 110, 20)})
    print("df1: \n", df1)
    cdf1 = DataFrame(df1)
    cdf2 = cdf1.drop_duplicates(ignore_index=True, env=env)
    print("duplicates dropped: \n", cdf2.df) if cdf2 else print("duplicates dropped: \n", cdf1.df)
    env.finalize()


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
drop_cuplicates()

# distributed join
# dist_concat()
