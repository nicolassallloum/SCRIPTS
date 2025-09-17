import pandas as pd

chunksize = 10**6  # 1M rows per chunk
for i, chunk in enumerate(pd.read_csv("Downloads/test_v8.csv", chunksize=chunksize)):
    chunk.to_csv(f"part_{i}.csv", index=False)
