
import argparse, sys
from datetime import datetime
from tron_gen import generate_tron_csv  # provided below

def main():
    # parse_known_args allows Jupyter's extra -f arg to be ignored safely
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=30_000_000)
    ap.add_argument("--out", type=str, default="tron_data.csv")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--gzip", action="store_true")
    ap.add_argument("--start-block", type=int, default=0)
    ap.add_argument("--start", type=str, default="2018-01-01T00:00:00Z")
    ap.add_argument("--progress-every", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=None)
    args, unknown = ap.parse_known_args()

    generate_tron_csv(
        rows=args.rows,
        out=args.out,
        shards=args.shards,
        gzip=args.gzip,
        start_block=args.start_block,
        start=args.start,
        progress_every=args.progress_every,
        workers=args.workers
    )

if __name__ == "__main__":
    main()
