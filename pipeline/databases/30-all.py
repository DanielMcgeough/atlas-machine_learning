#!/usr/bin/env python3
""" task 30 """


def list_all(mongo_collection):
    """  Lists all documents in a collection """
    return mongo_collection.find()
