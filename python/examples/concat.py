import cudf
import cupy as cp
import pygcylon as gc

def drop_cuplicates():
    env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)
    df1 = cudf.DataFrame({'first': cp.random.randint(100, 110, 20), 'second': cp.random.randint(100, 110, 20)})
    print("df1: \n", df1)
    cdf1 = gc.DataFrame(df1)
    cdf2 = cdf1.drop_duplicates(ignore_index=True, env=env)
    print("duplicates dropped: \n", cdf2.df) if cdf2 else print("duplicates dropped: \n", cdf1.df)
    env.finalize()


def dist_concat():
    env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = cudf.DataFrame({'first': cp.random.rand(6), 'second': cp.random.rand(6)})
    df2 = cudf.DataFrame({'second': cp.random.rand(6), 'first': cp.random.rand(6)})
    print(df1)
    print(df2)
    cdf1 = gc.DataFrame(df1)
    cdf2 = gc.DataFrame(df2)
    cdf3 = gc.concat([cdf1, cdf2], join="inner", env=env)
    print("distributed concated df:\n", cdf3.df)
    env.finalize()

def set_index():
    df1 = cudf.DataFrame({'first': cp.random.randint(100, 110, 20), 'second': cp.random.randint(100, 110, 20)})
    print("df1: \n", df1)
    cdf1 = gc.DataFrame(df1)
    cdf2 = cdf1.set_index("first")
    print("set index to first: \n")
    print(cdf2.df)
    cdf3 = cdf2.reset_index()
    print("index reset: \n", cdf3.df)


#####################################################
# local join test
# drop_cuplicates()

# distributed join
# dist_concat()

# set index
set_index()