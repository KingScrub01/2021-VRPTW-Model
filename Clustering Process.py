# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:51:03 2021

@author: Capta
"""

import sys
import numpy as np
import cplex
from cplex.exceptions import CplexError


def data_model():
    """Stores the data for the problem."""
    data = {}
    # Distance from node i to node j
    data['distance_matrix'] = [
        [
            0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354,
            468, 776, 662
        ],
        [
            548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674,
            1016, 868, 1210
        ],
        [
            776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164,
            1130, 788, 1552, 754
        ],
        [
            696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822,
            1164, 560, 1358
        ],
        [
            582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708,
            1050, 674, 1244
        ],
        [
            274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628,
            514, 1050, 708
        ],
        [
            502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856,
            514, 1278, 480
        ],
        [
            194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320,
            662, 742, 856
        ],
        [
            308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662,
            320, 1084, 514
        ],
        [
            194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388,
            274, 810, 468
        ],
        [
            536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764,
            730, 388, 1152, 354
        ],
        [
            502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114,
            308, 650, 274, 844
        ],
        [
            388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194,
            536, 388, 730
        ],
        [
            354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0,
            342, 422, 536
        ],
        [
            468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536,
            342, 0, 764, 194
        ],
        [
            776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274,
            388, 422, 764, 0, 798
        ],
        [
            662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730,
            536, 194, 798, 0
        ],
    ]
    data['num_vehicles'] = 4
    data['depot'] = 1
    data['demands'] = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]
    data['service_time'] = [1 for i in range(17)]
    data['vehicle_capacities'] = [25, 15, 15, 15]
    data['time_matrix'] = [
        [0, 6, 9, 8, 7, 3, 6, 2, 3, 2, 6, 6, 4, 4, 5, 9, 7],
        [6, 0, 8, 3, 2, 6, 8, 4, 8, 8, 13, 7, 5, 8, 12, 10, 14],
        [9, 8, 0, 11, 10, 6, 3, 9, 5, 8, 4, 15, 14, 13, 9, 18, 9],
        [8, 3, 11, 0, 1, 7, 10, 6, 10, 10, 14, 6, 7, 9, 14, 6, 16],
        [7, 2, 10, 1, 0, 6, 9, 4, 8, 9, 13, 4, 6, 8, 12, 8, 14],
        [3, 6, 6, 7, 6, 0, 2, 3, 2, 2, 7, 9, 7, 7, 6, 12, 8],
        [6, 8, 3, 10, 9, 2, 0, 6, 2, 5, 4, 12, 10, 10, 6, 15, 5],
        [2, 4, 9, 6, 4, 3, 6, 0, 4, 4, 8, 5, 4, 3, 7, 8, 10],
        [3, 8, 5, 10, 8, 2, 2, 4, 0, 3, 4, 9, 8, 7, 3, 13, 6],
        [2, 8, 8, 10, 9, 2, 5, 4, 3, 0, 4, 6, 5, 4, 3, 9, 5],
        [6, 13, 4, 14, 13, 7, 4, 8, 4, 4, 0, 10, 9, 8, 4, 13, 4],
        [6, 7, 15, 6, 4, 9, 12, 5, 9, 6, 10, 0, 1, 3, 7, 3, 10],
        [4, 5, 14, 7, 6, 7, 10, 4, 8, 5, 9, 1, 0, 2, 6, 4, 8],
        [4, 8, 13, 9, 8, 7, 10, 3, 7, 4, 8, 3, 2, 0, 4, 5, 6],
        [5, 12, 9, 14, 12, 6, 6, 7, 3, 3, 4, 7, 6, 4, 0, 9, 2],
        [9, 10, 18, 6, 8, 12, 15, 8, 13, 9, 13, 3, 4, 5, 9, 0, 9],
        [7, 14, 9, 16, 14, 8, 5, 10, 6, 5, 4, 10, 8, 6, 2, 9, 0]
    ]
    data['time_windows'] = [
        (0, 5),  # depot
        (7, 12),  # 1
        (10, 15),  # 2
        (16, 18),  # 3
        (10, 13),  # 4
        (0, 5),  # 5
        (5, 10),  # 6
        (0, 4),  # 7
        (5, 10),  # 8
        (0, 3),  # 9
        (10, 16),  # 10
        (10, 15),  # 11
        (0, 5),  # 12
        (5, 10),  # 13
        (7, 8),  # 14
        (10, 15),  # 15
        (11, 15),  # 16
    ]
    data['node_T_penalty'] = [1.0] * len(data['demands'])
    data['vehicle_T_penalty'] = [1.0] * data['num_vehicles']
    data['max_work_time'] = [8.0] * data['num_vehicles']
    data['fixed_vehicle_cost'] = [1, 1, 1, 1]
    
    return data

def clustering(dummy_vehicles):
    data = data_model()
    nbDepot = data['depot']
    nbDemand = len(data['demands'])
    C = []
    
    # Step 1(a)
    L = np.array([(i, data['time_windows'][i][0], data['time_windows'][i][1]) \
                     for i in range(nbDepot, nbDemand)],
                    dtype=[('index', int), ('ai', int), ('bi', int)])                
    L = np.sort(L, order=['ai', 'bi'])
    
    # Step 1(b)
    V = np.array([(i, data['vehicle_capacities'][i]/data['fixed_vehicle_cost'][i]) \
                  for i in range(data['num_vehicles'])],
                 dtype=[('index', int), ('ratio', float)])
    V = np.sort(V, order=['ratio'])[::-1]
    
    # Step 1(c)
    d_max = 1000
    delta = 0.5
    
    while len(L) > 0:
        # Checking for enough vehicles
        if len(C) >= data['num_vehicles'] and dummy_vehicles:
            data['num_vehicles'] += 1
            data['vehicle_capacities'].append(15)
            data['max_work_time'].append(8.0)
            data['fixed_vehicle_cost'].append(1)
        if len(C) >= data['num_vehicles'] and not dummy_vehicles:
            print('Not enough vehicles. Exiting with current clusters.')
            return C
            
        step = 2
        # Step 2-3(a)
        K=[L[0][0]]
        
        aC = data['time_windows'][K[0]][0]
        bC = data['time_windows'][K[0]][1]
        wC = data['demands'][K[0]]
        stC = data['service_time'][K[0]]
        
        # Step 3(b)
        L = np.delete(L, 0)
        L_prime = L
        
        # Step 4-6
        while len(L_prime) > 0:
            node_j = L_prime[0]
            # Step 4
            step = 4
            if wC + data['demands'][node_j[0]] <= data['vehicle_capacities'][len(C)]:
                
                # Step 5(a)
                d_min = data['distance_matrix'][node_j[0]][K[0]]
                for i in K:
                    d_ji = data['distance_matrix'][node_j[0]][i]
                    if d_ji <= d_min:
                        d_min = d_ji
                        node_i = i
                
                # Step 5(b)
                step = 5
                if d_ji <= d_max:
                    # Step 6
                    bj = node_j[2]
                    step = 6
                    if aC + stC + data['time_matrix'][node_j[0]][node_i] <= max(bC, bj):
                        # Step 7
                        step = 7
                        if aC + stC + data['time_matrix'][node_j[0]][node_i] \
                            + delta >= node_j[1]:
                                step = 8                           
                                # Step 8(a)
                                K.append(node_j[0])
                                wC += data['demands'][node_j[0]]
                                stC = max(stC + data['time_matrix'][node_j[0]][node_i] 
                                          + data['service_time'][node_j[0]], 
                                          node_j[1] + data['service_time'][node_j[0]] 
                                          - data['time_windows'][node_i][0])
                                
                                # Step 8(b)
                                if bC > node_j[2]:
                                    bC = node_j[2]
                                    
                                # node_j will be unique value as it has unique index No in its 0th index
                                L_prime = np.delete(L_prime, np.where(L_prime == node_j))
                                L = np.delete(L, np.where(L == node_j))
                                # print("L:", L)
                        else:        
                            L_prime = []
                            # print("C:", C)
                
            if len(L_prime) > 0:
                L_prime = np.delete(L_prime, 0)
        # Step 9
        # Prints last step before closing cluster
        print("Step:", step)
        C.append(K)
        # print("C:", C)    
    return C
        
        

        
if __name__ == '__main__': 
    clusters = clustering(dummy_vehicles=True)    
    print(clusters)
    
    
    
    
    
    