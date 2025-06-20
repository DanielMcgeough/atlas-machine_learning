#!/usr/bin/env python3
"""trying to work this is nearly impossible."""

if __name__ == '__main__':
    from requests import get

    API = "https://api.spacexdata.com/"
    LAUNCHES = "v4/launches/upcoming/"
    ROCKETS = "v4/rockets/"
    PADS = "v4/launchpads/"

    launches = get(API + LAUNCHES).json()

    # Find next rocket with required info
    launch = sorted(launches, key=lambda d: d['date_unix'])[0]

    # get launch and date
    name = launch['name']
    date = launch['date_local']

    # get rocket information
    rocket_info = get(API + ROCKETS + launch['rocket']).json()
    rocket_name = rocket_info['name']

    # get launchpad information
    pad_info = get(API + PADS + launch['launchpad']).json()
    pad_name = pad_info['name']
    pad_local = pad_info['locality']

    # print collected info
    print("{} ({}) {} - {} ({})".format(
        name, date, rocket_name, pad_name, pad_local))
