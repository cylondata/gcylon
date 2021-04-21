import cudf
import cupy as cp
from pygcylon import DataFrame, CylonEnv
from pygcylon.net.mpi_config import MPIConfig

def local_join():
    df1 = cudf.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    df2 = cudf.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    print("df1: \n", df1)
    print("df2: \n", df2)
    cdf1 = DataFrame(df1)
    cdf2 = DataFrame(df2)
    cdf3 = cdf1.join(cdf2)
    print("locally joined df: \n", cdf3.df)


def dist_join():
    env: CylonEnv = CylonEnv(config=MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = cudf.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    df2 = cudf.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    print(df1)
    print(df2)
    cdf1 = DataFrame(df1)
    cdf2 = DataFrame(df2)
    cdf3 = cdf1.join(other=cdf2, env=env)
    print("distributed joined df:\n", cdf3.df)
    env.finalize()

def test_mpi():
    env: CylonEnv = CylonEnv(config=MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)
    print("finalizing mpi")
    env.finalize()


#####################################################
# local join test
# local_join()

# distributed join
# dist_join()

test_mpi()