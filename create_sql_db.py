import os
import sqlite3
import pandas as pd

CSV_DIR = "./data"       # folder with csv files
DB_FILE = "toyota.db"        # output sqlite database file

def create_sqlite_from_csvs(csv_dir: str, db_file: str):
    # remove old db if exists
    if os.path.exists(db_file):
        os.remove(db_file)

    conn = sqlite3.connect(db_file)

    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            path = os.path.join(csv_dir, file)
            table_name = os.path.splitext(file)[0]

            print(f"Loading {file} -> table {table_name}")
            df = pd.read_csv(path)

            # write into SQL (replace if exists)
            df.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.close()
    print(f"Done. SQLite DB saved at: {db_file}")

if __name__ == "__main__":
    create_sqlite_from_csvs(CSV_DIR, DB_FILE)

