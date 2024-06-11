import pandas as pd
import numpy as np
import os
if os.path.basename(os.getcwd()) != 'food-pairing':
    os.chdir(os.path.dirname(os.getcwd()))

from data_loading import read_foods, read_molecules
from ml_utils import molecules2vec, find_n_neighbours
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec

def normalize(values):
    # values.insert(0, 0.0)
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    normalized_list = [(val - min_val) / range_val for val in values]
    return normalized_list

def find_cooccurences_pairings(df: pd.DataFrame, food: str):
    input_molecules = df.loc[df['food'] ==  food, 'foodb_ids'].iloc[0]
    df['similarity'] = df['foodb_ids'].apply(lambda x: len(set(x) & set(input_molecules)))
    similar_foods = df.sort_values(by='similarity', ascending=False).head(11)
    cooccurences_pairs = [x[0] for x in similar_foods[['food']].values.tolist()]
    similarity = [x[0] for x in similar_foods[['similarity']].values.tolist()]
    similarity = normalize(similarity)
    
    return cooccurences_pairs, similarity

def find_nn_pairings(df: pd.DataFrame, food: str):
    knn = NearestNeighbors(metric='manhattan', n_neighbors=10, n_jobs=-1)
    y = df['molecules_vector'].values
    knn.fit(y.tolist())
    entity_id = df.loc[df['food'] == food, 'food_id']
    raw_recommends, distances = find_n_neighbours(df, knn, int(entity_id.iloc[0]), n_neighbors=10)
    if np.asarray(raw_recommends).ndim > 1:
        raw_recommends = raw_recommends[0]
    if np.asarray(distances).ndim > 1:
        distances = distances[0]
    distances = normalize(distances)
    similarity = [1 - dist for dist in distances]
    return [str((df.iloc[x])['food']) for x in raw_recommends], similarity

def find_panther_pairings(food, G):
    pairings_dict = nx.panther_similarity(G, food, k=11, path_length=10, c=0.5, delta=0.1, eps=None, weight='weight')
    similarity = normalize(list(pairings_dict.values()))
    return pairings_dict.keys(), similarity

def find_node2vec_pairings(food, model):
    results = model.wv.most_similar(food, topn=10)
    node2vec_pairs = [pair[0] for pair in results]
    node2vec_distances = [pair[1] for pair in results]
    
    return node2vec_pairs, normalize(node2vec_distances)

def find_pairings(df: pd.DataFrame, food: str, G: nx.Graph, model):
    """Finds top 10 foods that share the most ingredient with given food"""

    # finding pairings by most cooccurences
    print("== Finding pairinigs by co-occurences ==")
    occurences_pairings, occurences_sim = find_cooccurences_pairings(df, food)
    print("== Finding pairinigs with NN ==")
    nn_pairings, nn_sim = find_nn_pairings(df, food)
    print("== Finding pairinigs by embeddings ==")
    node2vec_pairings, node2vec_sim = find_node2vec_pairings(food, model)
    print("== Finding pairinigs by Panther ==")
    panther_pairings, panther_sim = find_panther_pairings(food, G)

    return occurences_pairings, occurences_sim, nn_pairings, nn_sim, panther_pairings, panther_sim, node2vec_pairings, node2vec_sim

if __name__ == "__main__":
    G = nx.read_gexf("networks/food-cut.gexf", node_type=str, relabel=True)
    model = Word2Vec.load('output/node2vec')

    food_list = ['tomato', 'onion', 'cinnamon', 'pepper']
    for INPUT in food_list:
        food_df = molecules2vec(read_foods(), read_molecules())
        
        occurences_pairings, occurences_dist, nn_pairings, nn_dist, panther_pairings, panther_dist, node2vec_pairings, node2vec_sim = find_pairings(food_df, INPUT, G, model)

        result_df = pd.DataFrame(
            list(zip(occurences_pairings[1:], occurences_dist, nn_pairings, nn_dist, panther_pairings, panther_dist, node2vec_pairings, node2vec_sim)), 
            columns=['Cooccurences', 'Cooccurences Similarity',
                     'Nearest neighbors', 'Nearest neighbors Similarity',
                     'Panther', 'Panther Similarity',
                     'node2vec pairings', 'node2vec Similarity'])
        result_df.to_csv(f'results/{INPUT}_pairings.csv')
        print(f"Pairings for {INPUT} saved at \'results/{INPUT}_pairings.csv\'")
