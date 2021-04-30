import cupy as cp
import pygcylon as gc

def drop_cuplicates():
    env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)
    df1 = gc.DataFrame({'first': cp.random.randint(100, 110, 20), 'second': cp.random.randint(100, 110, 20)})
    print("df1: \n", df1)
    df2 = df1.drop_duplicates(ignore_index=True, env=env)
    print("duplicates dropped: \n", df2) if df2 else print("duplicates dropped: \n", df1)
    env.finalize()


def dist_concat():
    env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = gc.DataFrame({'first': cp.random.rand(6), 'second': cp.random.rand(6)})
    df2 = gc.DataFrame({'second': cp.random.rand(6), 'first': cp.random.rand(6)})
    print(df1)
    print(df2)
    df3 = gc.concat([df1, df2], join="inner", env=env)
    print("distributed concated df:\n", df3)
    env.finalize()

def set_index():
    df1 = gc.DataFrame({'first': cp.random.randint(100, 110, 20), 'second': cp.random.randint(100, 110, 20)})
    print("df1: \n", df1)
    df2 = df1.set_index("first")
    print("set index to first: \n")
    print(df2)
    df3 = df2.reset_index()
    print("index reset: \n", df3)


#####################################################
# drop duplicates test
drop_cuplicates()

# distributed join
# dist_concat()

# set index
# set_index()