#!/usr/bin/env python3
""" Task 34 """
from pymongo import MongoClient

template = """\
{} logs
Methods:
\tmethod GET: {}
\tmethod POST: {}
\tmethod PUT: {}
\tmethod PATCH: {}
\tmethod DELETE: {}
{} status check"""

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    db = client['logs']
    collection = db['nginx']

    logs = collection.count_documents({})
    GET = collection.count_documents({"method": "GET"})
    POST = collection.count_documents({"method": "POST"})
    PUT = collection.count_documents({"method": "PUT"})
    PATCH = collection.count_documents({"method": "PATCH"})
    DELETE = collection.count_documents({"method": "DELETE"})
    status = collection.count_documents({"method": "GET",
                                         "path": "/status"})

    print(template.format(logs, GET, POST, PUT, PATCH, DELETE, status))
