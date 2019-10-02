
import os
import json


def read_fr_JSON(filename):
    try:
         data = json.load(open(filename, 'r'))
         return data 
    except FileNotFoundError:
         print("JSON file not found.")
         pass

def write_to_JSON(data, filename):
    with open(filename, 'w') as f:
         json.dump(data, f, indent=4)

def tuple_to_string(triplet):
    # assumes form of (s,p,o)
    str = ','.join(triplet)
    return str

def string_to_tuple(triplet_string):
    # assumes triplet string is comma-separated (needed for JSON file)
    l = triplet_string.split(',')
    # generate tuple
    return tuple(l)
