import pandas as pd
from prefect import task

@task
def load_df(path):
    return pd.read_csv(path)