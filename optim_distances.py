"""Maximization of sum distance between pairs for a fair cultural exchange."""
import itertools
from typing import Dict, Tuple

import numpy as np
from geopy.distance import distance
from geopy.geocoders import Nominatim
from geopy.location import Location
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.preprocessing import MultiLabelBinarizer

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
# init all possible pairs, with its neg distances (min problem) and one hot encoded attendees
possible_pairs = list(itertools.combinations(ATTENDEES.keys(), 2))
pair_distances = np.array([get_neg_distance_between_points(ATTENDEES[a], ATTENDEES[b])
                           for a, b in possible_pairs])
attendee_matrix = MultiLabelBinarizer().fit_transform(possible_pairs)

# ****** formulation of optimization problem using mixed integer linear programming ******
# minimize the negative pair distances such that every attendee is included once:

# our decision variable is binary
bounds, integrality = Bounds(0,1), 1
# the transposed row wise sum of attendee matrix is equal to one
everybody_attends_once = LinearConstraint(attendee_matrix.T, lb=1, ub=1)
# solve
solution = milp(c=pair_distances, integrality=integrality, bounds=bounds,
                constraints=[everybody_attends_once])

# output result
idx_pairs = np.argwhere(solution.x == 1).flatten()
for idx_pair in idx_pairs:
    pair = "-".join(possible_pairs[idx_pair])
    distance = -pair_distances[idx_pair]
    print(f"{pair : <25}{int(distance)}km")
