"""Maximize sum distance for wichtel pairs without swaps for a fair cultural exchange."""
import itertools
from typing import Dict

import numpy as np
from geopy.distance import distance
from geopy.geocoders import Nominatim
from geopy.location import Location
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.preprocessing import OneHotEncoder

# get geo locations of attendees
geolocator = Nominatim(user_agent='wichteln')

ATTENDEES: Dict[str, Location] = {
    'Jochen': geolocator.geocode("Linsengericht"),
    'Sebastian H': geolocator.geocode("Etzelwang"),
    'Manuel': geolocator.geocode("Wolnzach"),
    'Sebastian K': geolocator.geocode("Appertshofen"),
    'Andreas': geolocator.geocode("Bamberg"),
    'Mike': geolocator.geocode("Limmer Alfeld"),
}

def get_neg_distance_between_points(loc_a: Location, loc_b: Location) -> float:
    """Calculate distance between two locations in km.

    Args:
        loc_a (Location): start
        loc_b (Location): end

    Returns:
        float: neg. distance [km]
    """
    _, lat_lon_a = loc_a
    _, lat_lon_b = loc_b
    return -distance(lat_lon_a, lat_lon_b).km

# ****** prepare the inputs ******
# init all possible permutations, with its neg distances (min problem)
# and one hot encoded givers and getters and unique pairs (a-b and b-a are unique)
possible_pairs = list(itertools.permutations(ATTENDEES.keys(), 2))
gives = [giver for giver, _ in possible_pairs]
gets = [getter for _, getter in possible_pairs]
unique_pairs = ["".join(sorted(giver + getter)) for giver, getter in possible_pairs]
pair_distances = np.array([get_neg_distance_between_points(ATTENDEES[a], ATTENDEES[b])
                           for a, b in possible_pairs])

ohc_attendees = OneHotEncoder(sparse=False)
ohc_pairs = OneHotEncoder(sparse=False)

give_matrix = ohc_attendees.fit_transform(np.array(gives)[:, None])
get_matrix = ohc_attendees.transform(np.array(gets)[:, None])
unique_pairs = ohc_pairs.fit_transform(np.array(unique_pairs)[:, None])

# ****** formulation of optimization problem using mixed integer linear programming ******
# minimize the negative distances such that every attendee gives and gets a beer
# and beer swaps are prevented

# our decision variable is binary
bounds, integrality = Bounds(0,1), 1

everybody_gives_one_beer = LinearConstraint(give_matrix.T, lb=1, ub=1)
everybody_gets_one_beer = LinearConstraint(get_matrix.T, lb=1, ub=1)
no_swaps = LinearConstraint(unique_pairs.T, lb=0, ub=1)

# solve
solution = milp(c=pair_distances, integrality=integrality, bounds=bounds,
                constraints=[everybody_gives_one_beer, everybody_gets_one_beer, no_swaps])

# output result
idx_pairs = np.argwhere(solution.x == 1).flatten()
for idx_pair in idx_pairs:
    pair = "-".join(possible_pairs[idx_pair])
    distance = -pair_distances[idx_pair]
    print(f"{pair : <25}{int(distance)}km")
