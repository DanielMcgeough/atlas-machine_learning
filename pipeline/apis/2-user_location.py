#!/usr/bin/env python3

"""This uses the github api
and is causing me psychosis."""

import sys
import requests
from datetime import datetime


def get_user_location(url):
    """finds a user"""
    response = requests.get(url)

    if response.status_code == 200:
        user_data = response.json()
        return user_data.get("location", "Location not provided")
    elif response.status_code == 404:
        return "Not found"
    elif response.status_code == 403:
        reset_timestamp = int(response.headers.get("X-Ratelimit-Reset", 0))

        reset_time = datetime.fromtimestamp(reset_timestamp)
        current_time = datetime.now()
        minutes_until_reset = (reset_time - current_time).seconds // 60
        return f"Reset in {minutes_until_reset} min"
    else:
        return f"Unexpected status code: {response.status_code}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
        sys.exit(1)

    url = sys.argv[1]
    location = get_user_location(url)
    print(location)
