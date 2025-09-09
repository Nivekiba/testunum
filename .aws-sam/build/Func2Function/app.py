import time

def lambda_handler(event, context):
    sleep_time_ms = event['sleep_time_ms']

    loops = event.get('loops', 30000000)  # number of iterations
    # Busy-spin loop to simulate delay without sleeping
    result = 0
    for i in range(loops):
        result += i * (i + 1)

    return {
        "loops": loops
    }
