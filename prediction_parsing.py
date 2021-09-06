# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 23:46:52 2021

@author: Yuseung Na
"""
from tqdm import tqdm


# data_prediction = open('C:/Ubuntu/GRIP/prediction_result/prediction_result.txt', 'r')
# data_prediction = data_prediction.read()

list_prediction = []
with open('C:/Ubuntu/GRIP/prediction_result/prediction_result.txt', 'r') as pred_file:
    for i in tqdm(range(398844), desc='prediction'):
        list_prediction.append((pred_file.readline()).split())
        
num = 0
final_num = 0
with open('C:/Ubuntu/GRIP/prediction_result/prediction_result.txt', 'r') as txt:
    prediction = txt.readlines()

for i in range(53):
    with open(f'prediction_result_{i}.txt', 'w') as txt:
        for j in tqdm(range(0, len(prediction))):
            if (j != 0) and (int(list_prediction[j+final_num][0]) + 10 < int(list_prediction[j+final_num-1][0])):
                final_num = num
                break
            txt.write(prediction[num])
            num = num + 1        
    

# for i in tqdm(range(len(prediction))):
#     if i > 1 and int(list_prediction[i][0]) > int(list_prediction[i-1][0]) + 10:
#         num += 1
    
#     with open(f'prediction_result_{num}.txt', 'w+') as txt:
#         txt.write(prediction[i])
    