
from .db_connector import MySQLConnector
import pandas as pd


def queryDB(query, labels = None):
    db = MySQLConnector()
    out, cols = db.execute(query)
    if labels is None:
        df = pd.DataFrame.from_records(out, columns = cols)
    else:
        df = pd.DataFrame.from_records(out, columns=labels)
    db.close()
    return df
