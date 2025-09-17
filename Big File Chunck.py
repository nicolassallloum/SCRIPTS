import pandas as pd
import paramiko
import io
import os

# --- CONFIG ---
SRC_HOST = "172.31.13.116"
SRC_USER = "cassandra"
SRC_PASS = "cassandra"   # or use keys

DST_HOST = "172.31.13.133"
DST_USER = "cassandra"
DST_PASS = "cassandra"   # or use keys

SRC_FILE = "/u01/cassandra/geo_scenarios_data.csv"
DST_DIR  = "/u01/kafka/csv-data"
BASENAME = "chunked_file_geo_"   # final files: chunked_file_geo_0.csv, ...

CHUNKSIZE = 10**6  # 1M rows per chunk
CSV_KW = dict(low_memory=False)  # tweak if needed (e.g., dtype, sep, encoding)

# --- Connect to source (read) ---
src_ssh = paramiko.SSHClient()
src_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
src_ssh.connect(SRC_HOST, username=SRC_USER, password=SRC_PASS)
sftp_src = src_ssh.open_sftp()

# Quick existence check on source file
try:
    sftp_src.stat(SRC_FILE)
except FileNotFoundError:
    sftp_src.close(); src_ssh.close()
    raise FileNotFoundError(f"Source file not found on {SRC_HOST}: {SRC_FILE}")

# --- Connect to destination (write) ---
dst_ssh = paramiko.SSHClient()
dst_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
dst_ssh.connect(DST_HOST, username=DST_USER, password=DST_PASS)
sftp_dst = dst_ssh.open_sftp()

# Ensure destination directory exists (mkdir -p style)
def ensure_remote_dir(sftp, path):
    parts = [p for p in path.split('/') if p]
    cur = '/'
    for p in parts:
        cur = os.path.join(cur, p)
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)

ensure_remote_dir(sftp_dst, DST_DIR)

rows_total = 0
with sftp_src.open(SRC_FILE, 'rb') as fh:  # file-like object for pandas
    for i, chunk in enumerate(pd.read_csv(fh, chunksize=CHUNKSIZE, **CSV_KW)):
        # Send chunk directly via putfo without saving to disk
        buff = io.StringIO()
        chunk.to_csv(buff, index=False)
        buff.seek(0)

        remote_name = f"{BASENAME}{i}.csv"
        remote_path = os.path.join(DST_DIR, remote_name)

        # upload from file-like object
        sftp_dst.putfo(buff, remote_path)
        rows_total += len(chunk)
        print(f"[OK] Wrote {remote_name} ({len(chunk)} rows) → {DST_HOST}:{remote_path}")

print(f"Done. Total rows processed: {rows_total}")

# Cleanup
sftp_src.close()
src_ssh.close()
sftp_dst.close()
dst_ssh.close()
