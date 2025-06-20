#!/usr/bin/env python3
"""This is brutal, how was I supposed
to have any idea how to do this."""
import requests
from sys import argv
import time


if __name__ == "__main__":
    try:
        r = requests.get(argv[1])

        # if user not found
        if r.status_code == 404:
            print('Not found')

        # print remaining time if rate limit exceeded
        if r.status_code == 403:
            now = int(time.time())
            reset = int(r.headers['X-RateLimit-Reset'])
            diff = reset - now
            print("Reset in {} min".format(diff // 60))

        # print location if no error
        if r.status_code == 200:
            print(r.json()['location'])

    # Print exception if incorrect URL is passed
    except requests.exceptions.MissingSchema as ms:
        print(ms)
