import pandas as pd
import ast
   
def string_to_list(string: str) -> list:
    try:
        return ast.literal_eval(string)
    except:
        return [s.strip("'") for s in string[1:-1].split(', ')]


def read_foods(source: str = None) -> pd.DataFrame:
    if source == "flavordb":
        food_df = pd.read_csv(
            "data/flavordb_foods_filtered.csv", 
            sep=';', 
            index_col=False
            )
        food_df['synonyms'] = food_df['synonyms'].apply(string_to_list)
        food_df['foodb_ids'] = food_df['foodb_ids'].apply(string_to_list)
    elif source == 'foodb':
        food_df = pd.read_csv(
            "data/foodb_foods_filtered.csv", 
            sep=';', 
            index_col=False
            )

        food_df['foodb_ids'] = food_df['foodb_ids'].apply(string_to_list)
        food_df['quantities'] = food_df['quantities'].apply(string_to_list)
    
    elif source == 'reduced':
        food_df = pd.read_csv(
            "data/food_reduced.csv", 
            sep=';', 
            index_col=False
            )
        food_df['synonyms'] = food_df['synonyms'].apply(string_to_list)
        food_df['foodb_ids'] = food_df['foodb_ids'].apply(string_to_list)
    else:
        food_df = pd.read_csv(
            "data/food_cut.csv", 
            sep=';', 
            index_col=False
            )
        food_df['synonyms'] = food_df['synonyms'].apply(string_to_list)
        food_df['foodb_ids'] = food_df['foodb_ids'].apply(string_to_list)
        # food_df['quantities'] = food_df['quantities'].apply(string_to_list)
    
    food_df['molecules'] = food_df['molecules'].apply(string_to_list)

    return food_df
    
    
def read_molecules(source: str = 'flavordb') -> pd.DataFrame:
    if source == "flavordb":
        molecules_df = pd.read_csv(
            "data/flavordb_molecules_cut.csv", 
            sep=';',
            index_col=False
            )
        molecules_df['flavors'] = molecules_df['flavors'].apply(string_to_list)
    else:
        molecules_df = pd.read_csv(
            "data/foodb_molecules_filtered.csv",
            sep=';',
            index_col=False,
        )
    return molecules_df


def read_recipes():
    recipe_df = pd.read_csv('data/ingredients.csv', sep=';', index_col=None)
    recipe_df['IDs'].apply(string_to_list)
    return recipe_df
    