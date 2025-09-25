from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from collections import Counter
import pandas as pd
import time

def main():
    start_time = time.time()  # Start timing

    try:
        # Step 1: Connect to cluster
        cluster = Cluster(['172.31.13.116'], port=9042)
        session = cluster.connect()

        # Step 2: Check Cassandra version
        version_row = session.execute("SELECT release_version FROM system.local").one()
        print(f"[INFO] Connected to Cassandra version: {version_row.release_version}")

        # Step 3: Set keyspace
        session.set_keyspace('trc')

        # Step 4: Query with paging
        fetch_size = 500
        query = SimpleStatement("SELECT from_address FROM trc20_crypto_transfers", fetch_size=fetch_size)

        print("[INFO] Executing query...")
        rows = session.execute(query)

        # Step 5: Count addresses
        address_counter = Counter()
        count = 0
        for row in rows:
            address = row.from_address
            if address:
                address_counter[address] += 1
                count += 1
                if count % 10000 == 0:
                    print(f"[INFO] Processed {count} rows...")

        print(f"[INFO] Finished processing {count} rows.")

        # Step 6: Export to CSV
        print("[INFO] Writing results to 'address_counts.csv'...")
        df = pd.DataFrame(address_counter.items(), columns=["from_address", "transfer_count"])
        df.sort_values(by="transfer_count", ascending=False, inplace=True)
        df.to_csv("address_counts.csv", index=False)
        print("[INFO] Export complete: 'address_counts.csv' created.")

        # Step 7: Print top 10
        print("\nTop 10 most frequent 'from_address' values:")
        for address, count in address_counter.most_common(10):
            print(f"{address}: {count} transfers")

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        try:
            cluster.shutdown()
        except:
            pass

        # Step 8: Show total processing time
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\n?? Total processing time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
