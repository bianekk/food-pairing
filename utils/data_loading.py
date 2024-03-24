import pandas as pd
import ast

def read_compounds():
    compounds_df = pd.read_csv("foodb/Content.csv")
    compounds_df['orig_food_common_name'] = compounds_df['orig_food_common_name'].astype(str)
    compounds_df['orig_food_common_name'] = compounds_df['orig_food_common_name'].apply(lambda x: x.replace('(', '').replace(')', ''))
    result_df = compounds_df.groupby('orig_food_common_name')['source_id'].agg(list).reset_index()

    return compounds_df, result_df

    
def string_to_list(string):
    return ast.literal_eval(string)


def read_food_molecules(source: str = "flavordb"):
    if source == "flavordb":
        flavor_df = pd.read_csv(
            "data/flavordb_v4.csv", 
            sep=';', 
            index_col=False
            )
        flavor_df['synonyms'] = flavor_df['synonyms'].apply(string_to_list)
    else:
        flavor_df = pd.read_csv(
            "data/foodb.csv", 
            sep=';', 
            index_col=False
            )

        flavor_df['food_id'] = flavor_df['food_id'].astype(float)
        flavor_df['food_id'] = flavor_df['food_id'].astype(int)
    
    flavor_df['molecules'] = flavor_df['molecules'].apply(string_to_list)

    return flavor_df
    
    
def read_molecules_flavors():
    molecules_df = pd.read_csv(
        "data/molecules.csv", 
        index_col=False
        )
    molecules_df['flavors'] = molecules_df['flavors'].apply(string_to_list)

    return molecules_df
    