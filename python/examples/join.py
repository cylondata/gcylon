import cudf
import cupy as cp
import pygcylon

def local_join():
    df1 = cudf.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    df2 = cudf.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    print("df1: \n", df1)
    print("df2: \n", df2)
    cdf1 = pygcylon.DataFrame(df1)
    cdf2 = pygcylon.DataFrame(df2)
    cdf3 = cdf1.join(cdf2)
    print("locally joined df: \n", cdf3.df)
    print("******************** df1 column Names: \n", list(df1._data.keys()))


def dist_join():
    env: pygcylon.CylonEnv = pygcylon.CylonEnv(config=pygcylon.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = cudf.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    df2 = cudf.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    print(df1)
    print(df2)
    cdf1 = pygcylon.DataFrame(df1)
    cdf2 = pygcylon.DataFrame(df2)
    cdf3 = cdf1.join(other=cdf2, env=env)
    print("distributed joined df:\n", cdf3.df)
    env.finalize()

def test_mpi():
    env: pygcylon.CylonEnv = pygcylon.CylonEnv(config=pygcylon.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)
    print("finalizing mpi")
    env.finalize()


#####################################################
# local join test
local_join()

# distributed join
dist_join()

# test_mpi()