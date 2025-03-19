import pandas as pd
import os
directory_list = list()
for root, dirs, files in os.walk("Data/Images", topdown=False):
    for name in dirs:
        directory_list.append(name)
pokemonStatistics = pd.read_csv('Data/Pokemon.csv')
num1 = 0
listofFolders = list()
for index,row in pokemonStatistics.iterrows():
    val = row['Name']
    if val in directory_list:
        listofFolders.append("Data/Images/"+val)
    else:
        pokemonStatistics.drop(index, inplace=True)
pokemonStatistics["ImagesFolderLocation"] = listofFolders

print(pokemonStatistics['ImagesFolderLocation'])