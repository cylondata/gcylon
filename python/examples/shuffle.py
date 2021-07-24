import cupy as cp
import pygcylon as gcy

env: gcy.CylonEnv = gcy.CylonEnv(config=gcy.MPIConfig(), distributed=True)
print("CylonContext Initialized: My rank: ", env.rank)

start = 100 * env.rank
df = gcy.DataFrame({'first': cp.random.randint(start, start + 10, 10),
                   'second': cp.random.randint(start, start + 10, 10)})
print("initial df from rank: ", env.rank, "\n", df)

shuffledDF = df.shuffle(on="first", ignore_index=True, env=env)

print("shuffled df from rank: ", env.rank, "\n", shuffledDF)

env.finalize()
print("after finalize from the rank:", env.rank)
