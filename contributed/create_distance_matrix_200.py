import csv
import numpy as np
import pandas as pd

matrix_size = 50

with open('/mnt/drive/Amir/kbtg_results/model3/50/output_all.csv', 'r') as file:
    reader = csv.reader(file)
    list = list(reader)

# print(list)

#Convert list of tuples to dataframe and set column names and indexes
df = pd.DataFrame(list, columns = ['GroundTruth' , 'Distance', 'Prediction'])
df = df.sort_values(['GroundTruth', 'Prediction'])
# print(df)

df_to_list = df.values.tolist()

matrix = np.zeros((matrix_size,matrix_size))

for i in range(0, matrix_size):
    counter = matrix_size * (i)
    for j in range(0, matrix_size):
        # print(i,j, list[counter + j][1], counter + j)
        matrix[i][j] = df_to_list[counter + j][1]


print(matrix)