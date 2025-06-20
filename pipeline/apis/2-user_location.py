#!/usr/bin/env python3
"""Task 2. Rate me is you can!"""
import requests
import sys
from datetime import datetime


def get_user_location(url):
    """script that prints the location of a specific user"""
    response = requests.get(url)

    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_timestamp = response.headers.get('X-Ratelimit-Reset')
        reset_time = datetime.fromtimestamp(int(reset_timestamp))
        wait_time = reset_time - datetime.now()
        minutes = divmod(wait_time.total_seconds(), 60)[0]
        print("Reset in {} min".format(int(minutes)))
    else:
        user_data = response.json()
        print(user_data.get('location', "No location found"))


if __name__ == '__main__':
    get_user_location(sys.argv[1])
