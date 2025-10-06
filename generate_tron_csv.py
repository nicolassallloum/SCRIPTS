#!/usr/bin/env python3
"""
Generate large synthetic TRON-like CSV data.

Columns:
block,txid,timestamp,amount,contract_address,currency,event_name,from_address,to_address

Usage examples:
  # Single file, 30M rows
  python generate_tron_csv.py --rows 30000000 --out tron_data.csv

  # Sharded (8 files x 3.75M each), faster on multi-core
  python generate_tron_csv.py --rows 30000000 --shards 8 --out tron_data_part.csv

  # Gzipped shards
  python generate_tron_csv.py --rows 30000000 --shards 8 --gzip --out tron_data_part.csv.gz

  # Control starting block/timestamp
  python generate_tron_csv.py --rows 1000000 --start-block 36000000 --start '2018-01-01T00:00:00Z'
"""
import argparse, csv, gzip, math, os, random, string, sys, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Tuple

CURRENCIES = ["TRX","USDT","USDC","BTT","WIN","NFT"]
EVENTS = ["Transfer","TransferSingle","TransferBatch","Approval","Mint","Burn"]

HEX = "0123456789abcdef"
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def rand_hex(n:int)->str:
    return "".join(random.choice(HEX) for _ in range(n))

def rand_txid()->str:
    return rand_hex(64)

def rand_contract()->str:
    # 0x-prefixed EVM-like address (42 chars)
    return "0x" + rand_hex(40)

def rand_tron_base58()->str:
    # Looks like T... base58 (34 chars typical)
    return "T" + "".join(random.choice(B58) for _ in range(33))

def iter_rows(start_block:int, start_dt:datetime, n:int)->Iterator[Tuple]:
    block = start_block
    dt = start_dt
    # Use modest time step to vary timestamps
    step = timedelta(seconds=3)
    for i in range(n):
        # Occasionally advance block more than 1 to create variety
        inc = 1 if (i % 5) else random.randint(1, 3)
        block += inc
        dt += step
        txid = rand_txid()
        amount = random.randint(1, 10_000_000)  # integer (sun-like) amount
        contract = rand_contract()
        currency = random.choice(CURRENCIES)
        event = random.choices(EVENTS, weights=[70,10,5,10,3,2], k=1)[0]
        from_addr = rand_tron_base58()
        to_addr = rand_tron_base58()
        # ISO8601 UTC
        ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        yield (block, txid, ts, amount, contract, currency, event, from_addr, to_addr)

def write_shard(path:str, rows:int, start_block:int, start_dt:datetime, gzip_out:bool)->int:
    t0 = time.time()
    if gzip_out:
        f = gzip.open(path, "wt", newline="")
    else:
        f = open(path, "w", newline="")
    with f:
        w = csv.writer(f)
        w.writerow(["block","txid","timestamp","amount","contract_address","currency","event_name","from_address","to_address"])
        for row in iter_rows(start_block, start_dt, rows):
            w.writerow(row)
    return int(time.time()-t0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True, help="Total rows to generate")
    ap.add_argument("--out", type=str, default="tron_data.csv", help="Output file (or prefix for shards)")
    ap.add_argument("--shards", type=int, default=1, help="Number of shards (files) to split rows into")
    ap.add_argument("--gzip", action="store_true", help="Write gzip-compressed CSV(s)")
    ap.add_argument("--start-block", type=int, default=0, help="Starting block number")
    ap.add_argument("--start", type=str, default="2018-01-01T00:00:00Z", help="Start timestamp (ISO8601, UTC)")
    args = ap.parse_args()

    if args.rows <= 0:
        print("rows must be > 0", file=sys.stderr); sys.exit(1)
    if args.shards <= 0:
        print("shards must be > 0", file=sys.stderr); sys.exit(1)

    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        print("Invalid --start format; use e.g. 2018-01-01T00:00:00Z", file=sys.stderr)
        sys.exit(1)

    rows_per = args.rows // args.shards
    remainder = args.rows % args.shards

    print(f"[INFO] Generating {args.rows:,} rows into {args.shards} shard(s) ...")
    print(f"[INFO] Base file: {args.out}  | gzip={args.gzip}")
    total_written = 0
    start_block = args.start_block
    start_dt_local = start_dt

    for i in range(args.shards):
        n = rows_per + (1 if i < remainder else 0)
        if n == 0:
            continue
        if args.shards == 1:
            out_path = args.out
        else:
            # e.g., tron_data_part.csv -> tron_data_part_00.csv
            base, ext = (args.out, "")
            if "." in args.out:
                base = args.out[:args.out.rfind(".")]
                ext = args.out[args.out.rfind("."):]
            out_path = f"{base}_{i:02d}{ext}"
        t = write_shard(out_path, n, start_block, start_dt_local, args.gzip)
        total_written += n
        # Advance starting points for next shard a bit to avoid overlap
        start_block += n + 1000
        start_dt_local += timedelta(seconds=3 * n + 60)
        print(f"[OK] Shard {i+1}/{args.shards} -> {out_path}  rows={n:,}  took~{t}s")

    print(f"[DONE] Wrote {total_written:,} rows total.")

if __name__ == "__main__":
    # Slightly deterministic randomness for repeatability; remove for full randomness
    random.seed(42)
    main()
