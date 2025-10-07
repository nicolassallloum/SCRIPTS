# pip install cassandra-driver
from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent_with_args
from cassandra.query import SimpleStatement
from decimal import Decimal

KEYSPACE = "ledger"
TABLE = "events"
COL_SRC = "amount"        # text column
COL_DST = "amount_num"    # numeric column you added
PAGE_SIZE = 5000
CONCURRENCY = 64

def parse_amount(s):
    if s is None or s == "":
        return None
    # Choose ONE depending on your target type:
    # return int(s)                 # for bigint/varint if the string is an integer
    return Decimal(s)               # for decimal (handles 18-decimals tokens, etc.)

cluster = Cluster(["127.0.0.1"])   # adjust
session = cluster.connect(KEYSPACE)
session.default_fetch_size = PAGE_SIZE

# 1) Prepare statements
select_stmt = SimpleStatement(
    f"SELECT txid, {COL_SRC} FROM {TABLE}",
    fetch_size=PAGE_SIZE
)
# Use the table’s PRIMARY KEY instead of txid alone if txid isn’t the full PK!
update_stmt = session.prepare(
    f"UPDATE {TABLE} SET {COL_DST} = ? WHERE txid = ?"
)

fetched = 0
updated = 0
for page in session.execute(select_stmt):
    rows = list(page.current_rows)
    args = []
    for r in rows:
        try:
            val = parse_amount(r.amount)
        except Exception:
            val = None  # or log/skip bad rows
        if val is not None:
            args.append((val, r.txid))
    if args:
        results = execute_concurrent_with_args(session, update_stmt, args, concurrency=CONCURRENCY)
        updated += sum(1 for (success, _) in results if success)
    fetched += len(rows)
    if page.has_more_pages:
        continue

print(f"Scanned={fetched}, Updated={updated}")
cluster.shutdown()
