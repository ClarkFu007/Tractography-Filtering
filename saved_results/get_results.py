import os

import numpy as np

def get_results(folder_name):
    print("Loading from ", folder_name)
    results_list = os.listdir(folder_name)
    results = np.zeros((2*len(results_list), 1), dtype='float')
    for index_i, results_i in enumerate(results_list):
        filename = os.path.join(folder_name, results_i)
        results[index_i*2: index_i*2+2, 0] = np.load(filename)
    print(folder_name + ":")
    for index_i in range(results.shape[0]):
        print(results[index_i, 0])

    
if __name__ == '__main__':
    get_results(folder_name='motor_sensory_normal_0.7_10_10')
    get_results(folder_name='motor_sensory_normal_0.8_10_10')
    get_results(folder_name='motor_sensory_sphere_0.7_10_10')
    get_results(folder_name='motor_sensory_sphere_0.8_10_10')