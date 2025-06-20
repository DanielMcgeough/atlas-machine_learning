#!/usr/bin/env python3
"""Task 0"""
import requests


def availableShips(passengerCount):
    """
    Using the Swapi API, returns the list of ships that can hold a given
    number of passengers
    """
    page = 1
    api = "https://swapi-api.alx-tools.com/api/"
    category = "starships/"
    ships = []

    while True:
        r = requests.get(api + category + "?page={}".format(page)).json()

        for ship in r['results']:
            passengers = ship['passengers']
            try:
                if int(passengers.replace(',', '')) >= passengerCount:
                    ships.append(ship['name'])
            except ValueError:
                pass

        if not r['next']:
            break

        page += 1

    return ships
