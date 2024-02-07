from prefect import task, flow
from prefect.concurrency.sync import concurrency
import time

@task
def process(x,y):
    with concurrency("test", occupy=1):
        time.sleep(5)
        return x+y

@flow
def my_flow():
    for x, y in zip(range(10), range(10,20)):
        process.submit(x,y)
if __name__ == '__main__':
    my_flow()
