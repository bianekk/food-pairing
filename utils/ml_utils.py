import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import os
if os.path.basename(os.getcwd()) != 'food-pairing':
    os.chdir(os.path.dirname(os.getcwd()))

from utils.data_loading import read_food_molecules
import matplotlib.pyplot as plt
import plotly.express as px

def pad_lists(lst, max_len: int = 10):
    """
    Function to even lists of molecules contained in the food databases. If a list is shorter than max_len, the missing values will be filled with zeros.
    On the other hand, a longer list will be cut down to max_len molecules, without changing the order. 
    Args:
        lst - list of molecules
        max_len - length of the output lists, default 10
    """
    out = [0] * max_len
    if len(lst) == max_len:
        lst = [*map(int, lst)]
        return np.asarray(lst)
    elif len(lst) < max_len:
        out[:len(lst)] = lst
        out = [*map(int, out)]
        return np.asarray(out)
    else:
        lst = [*map(int, lst)]
        return np.asarray(lst[:max_len])
    

def find_n_neighbours(df, model, target_id, n_neighbors=10):
    """
    Given a fit NN model, find the nearest neighbor of a given entity.
    Args:
        df - the dataframe, that was used in the model (FooDB/FlavorDB dataframe)
        model - fit NN model
        target_id - the ID of the target entity (not the same as the df index)
        n_neighbors - number of similar instances to return, default = 10
    """
    target_id_vec = df.loc[df['food_id'] == target_id, 'molecules']

    if len(target_id_vec) == 0: 
        return [0], [0]
    
    target_id_vec = target_id_vec.values.tolist()

    distances, indices = model.kneighbors(
        target_id_vec,
        n_neighbors=n_neighbors+1
        )

    reccomends = indices.squeeze().tolist()
    distances = distances.squeeze().tolist()
    
    return reccomends[1:], distances[1:]
