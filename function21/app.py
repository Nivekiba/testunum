import time

def lambda_handler(event, context):
    sleep_time_ms = event['sleep_time_ms']

    time.sleep(sleep_time_ms/1000)

    return {
        "sleep_time_ms": sleep_time_ms
    }
