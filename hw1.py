#
# CS 196 Data Hackerspace
# Assignment 1: Data Parsing and NumPy
# Due October 1st, 2018
#

import json
import csv
import numpy as np
import pandas as pd


def histogram_times(filename):
    dataset = pd.read_csv(filename)
    valid_hour = []
    for time in dataset['Time']:
        if time is np.nan:
            continue

        value = time.split(':')
        if len(value) >= 2:     # some have format as "c: 1:00"
            hour = value[-2]
            if not hour[0].isdigit(): # some have format as "c16:50"
                hour = hour[1:]
            if int(hour) > 23: # ignore an error with hour = 114
                continue
            valid_hour.append(int(hour))
    
    hist, bins = np.histogram(valid_hour, bins=list(range(24)))
    return hist


def weigh_pokemons(filename, weight):
    dataset = None
    with open(filename, 'r') as f:
        dataset = json.load(f)
    pokemons = dataset['pokemon']

    names = []
    for poke in pokemons:
        w = float(poke['weight'][:-2])
        if w == weight:
            names.append(poke['name'])
    return names


def single_type_candy_count(filename):
    dataset = None
    with open(filename, 'r') as f:
        dataset = json.load(f)
    pokemons = dataset['pokemon']

    count = 0
    for pokemon in pokemons:
        if 'candy_count' in pokemon and len(pokemon['type']) == 1:
            count += int(pokemon['candy_count'])
    return count


def reflections_and_projections(points):
    # Reflects the point over the line y = 1
    points[0, :] = 2 - points[0, :]

    # Rotates the point pi/2 radians around the origin
    theta = np.pi/2
    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points = np.dot(rotate_matrix, points)

    # Projects the point onto the line y = 3x
    m = 3
    project_matrix = np.array([[1, m], [m, m*m]])/(m*m + 1)
    points = np.dot(project_matrix, points)

    return points


def normalize(image):
    p_min = np.min(image)
    p_max = np.max(image)
    return 255*(image - p_min)/(p_max - p_min)


def sigmoid_normalize(image, a):
    return 255/(1 + np.exp(-(image-128)/a))