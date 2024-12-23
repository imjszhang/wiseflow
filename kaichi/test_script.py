import sqlite3

def read_database(db_path):
    # 连接到 SQLite 数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        print(f"Table: {table_name}")

        # 查询表中的所有数据
        cursor.execute(f"SELECT * FROM {table_name};")
        rows = cursor.fetchall()

        for row in rows:
            print(row)

    # 关闭连接
    conn.close()

if __name__ == "__main__":
    db_path = "core/pb/pb_data/data.db"  # 数据库路径
    read_database(db_path)