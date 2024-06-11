import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import os
if os.path.basename(os.getcwd()) != 'food-pairing':
    os.chdir(os.path.dirname(os.getcwd()))

import plotly.express as px
import plotly.io as pio

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
    

def find_n_neighbours(molecules_vectors, model, target_id, n_neighbors=10):
    """
    Given a fit NN model, find the nearest neighbor of a given entity.
    Args:
        molecules_vectors -- the dataframe, that was used in the model (FooDB/FlavorDB dataframe) OR np.array of embedding inputs
        model - fit NN model
        target_id - the ID of the target entity (not the same as the df index)
        n_neighbors - number of similar instances to return, default = 10
    """
    if type(molecules_vectors) != np.ndarray:
        target_id_vec = molecules_vectors.loc[molecules_vectors['food_id'] == target_id, 'molecules_vector']
        target_id_vec = target_id_vec.values.tolist()
    else:
        target_id_vec = molecules_vectors[target_id]
        target_id_vec = [target_id_vec]
    if len(target_id_vec) == 0: 
        return [0], [0]
    
    distances, indices = model.kneighbors(
        target_id_vec,
        n_neighbors=n_neighbors+1
        )

    reccomends = indices.squeeze().tolist()
    distances = distances.squeeze().tolist()
    
    return reccomends[1:], distances[1:]

def plot_reduction(df, embedding, name):
    fig = px.scatter(
        embedding, 
        x=0, y=1, 
        color=df['category'].values,
        hover_data=[df['food'].values],
        labels={'hover_data_0':'food',
                'color': 'category',
                },
    #text=flavor_df['food']
    )
    # fig.update_traces(textposition='top center')

    fig.update_layout(
            font=dict(
                family="CMU Serif",
                size=14, 
            )
        )
        
    fig.update_layout( 
        template = 'ggplot2', 
        height=500,
        width = 900,
        margin=dict(l=20, r=20, t=20, b=20),
        # title_text='Visualization by  UMAP'
    )
    config = {
    'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'height': 600,
        'width': 900,
        'scale':6 # Multiply title/legend/axis/canvas sizes by this factor
    }
    }
    
    fig.show(config=config)
    pio.write_image(fig, f"images/{name}.png", scale=6, width=900, height=500)


def molecules2vec(food_df, molecule_df):
    def list2onehot(lst):
        molecules_vec = [0] * len(all_molecules)
        for i in range(len(all_molecules)):
            if all_molecules[i] in lst:
                molecules_vec[i] = 1
            else:
                molecules_vec[i] = 0
        
        return molecules_vec
    
    all_molecules = molecule_df['foodbid'].values.tolist()
    food_df['molecules_vector'] = food_df['foodb_ids'].apply(list2onehot)

    return food_df
