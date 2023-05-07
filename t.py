import time
from sqlitedict import SqliteDict


if __name__ == "__main__":
    d = SqliteDict("./my_db.sqlite")

    d["key1"] = {}

    for i in range(10):
        d[f"key{i}"] = {str(i): i}
        time.sleep(.4)

    d.commit()

    for i in range(10):
        print(d[f"key{i}"])