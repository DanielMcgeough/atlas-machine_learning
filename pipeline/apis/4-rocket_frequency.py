#!/usr/bin/env python3
""""""

if __name__ == '__main__':
    from requests import get

    API = 'https://api.spacexdata.com/v4/'

    rocket_names = {}   # (rocket ID: rocket name)
    rockets_count = {}  # {rocket name: count}

    # get all launch records from API
    launches = get(API + 'launches/').json()
    for launch in launches:
        rocket_id = launch['rocket']    # get rocket ID for launch

        # get rocket name from rocket_names
        try:
            name = rocket_names[rocket_id]  # get rocket name
            rockets_count[name] += 1        # increment count

        # rocket name doesn't yet exist in rocket_names
        except KeyError:
            # get rocket info from API
            name = get(API + 'rockets/' + rocket_id).json()['name']

            rocket_names[rocket_id] = name  # add ID to name dict
            rockets_count[name] = 1         # add name to count dict

    # sort alphabetically into list of tuples
    rockets_count = sorted(rockets_count.items(), key=lambda d: d[0])

    # sort list by value and print
    for rocket in sorted(rockets_count, key=lambda d: d[1], reverse=True):
        print('{}: {}'.format(rocket[0], rocket[1]))
