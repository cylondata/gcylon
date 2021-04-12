import cudf

inputFile = "data/cities.csv"
tbl = cudf.read_csv(inputFile)
tbl.info()

from pygcylon import simple
simple.rows(tbl)
