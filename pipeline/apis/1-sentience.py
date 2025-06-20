#!/usr/bin/env python3
"""Isn't Kashyyk so much fun to say.
The trees are called Wroshyr I think."""
import requests


def sentientPlanets():
    """
    Using the Swapi API, returns the list of ships that can hold a given
    number of passengers
    """
    page = 1
    api = "https://swapi-api.alx-tools.com/api/"
    category = "species/"
    planets = []

    while True:
        r = requests.get(api + category + "?page={}".format(page)).json()
        for species in r['results']:
            classification = species['classification']
            designation = species['designation']

            if classification or designation == 'sentient':
                try:
                    r2 = requests.get(species['homeworld']).json()
                    homeworld = r2['name']
                    if homeworld not in planets:
                        planets.append(homeworld)

                except requests.exceptions.MissingSchema:
                    pass

        page += 1
        if not r['next']:
            break

    return list(planets)
