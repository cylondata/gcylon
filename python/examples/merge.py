import cupy as cp
import pygcylon as gc

def local_merge():
    df1 = gc.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})
    df2 = gc.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})
    print("df1: \n", df1)
    print("df2: \n", df2)
    df3 = df1.merge(right=df2, how="left", on="first", left_index=False, right_index=False)
    print("locally merged df: \n", df3)


def dist_merge():
    env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = gc.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})
    df2 = gc.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})

    print(df1)
    print(df2)
    df3 = df1.merge(right=df2, on="first", how="left", left_on=None, right_on=None, left_index=False, right_index=False, env=env)
    print("distributed joined df:\n", df3)
    env.finalize()

#####################################################
# local join test
local_merge()

# distributed join
# dist_merge()
