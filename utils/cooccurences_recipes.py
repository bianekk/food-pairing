import pandas as pd
import numpy as np
import os
if os.path.basename(os.getcwd()) != 'food-pairing':
    os.chdir(os.path.dirname(os.getcwd()))

from data_loading import read_recipes


def find_cooccurences_pairings(df: pd.DataFrame, food: str):
    input_molecules = df.loc[df['Ingredient'] ==  food, 'IDs'].iloc[0]
    df['similarity'] = df['IDs'].apply(lambda x: len(set(x) & set(input_molecules)))
    similar_foods = df.sort_values(by='similarity', ascending=False).head(11)
    
    return similar_foods[['Ingredient']].values.tolist()


if __name__ == "__main__":
    INPUT = 'green peppers'

    recipe_df = read_recipes()
    occurences_pairings = find_cooccurences_pairings(recipe_df, INPUT)

    result_df = pd.DataFrame(
        list(zip(occurences_pairings[1:])), 
        columns=['Cooccurences'])
    result_df.to_csv(f'results/{INPUT}_recipe_pairings.csv')
    print(f"Pairings for {INPUT} saved at \'results/{INPUT}_recipe_pairings.csv\'")
