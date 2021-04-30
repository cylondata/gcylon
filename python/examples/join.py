import cupy as cp
import pygcylon as gc

def local_join():
    df1 = gc.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    df2 = gc.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    print("df1: \n", df1)
    print("df2: \n", df2)
    df3 = df1.join(df2)
    print("locally joined df: \n", df3)


def dist_join():
    env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = gc.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    df2 = gc.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
    print(df1)
    print(df2)
    df3 = df1.join(other=df2, env=env)
    print("distributed joined df:\n", df3)
    env.finalize()

def test_mpi():
    env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)
    print("finalizing mpi")
    env.finalize()


#####################################################
# local join test
local_join()

# distributed join
dist_join()

# test_mpi()