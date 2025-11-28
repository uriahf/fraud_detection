import polars as pl


def read_data(file_path="data/creditcard.csv"):
    return pl.read_csv(file_path, schema_overrides={"Time": pl.Float64})
