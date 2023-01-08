import pickle
import numpy as np
from copy import deepcopy
from itertools import combinations

def preprocess_symbol_vector(points):    
    
    M = 20

    # You do not have labels here beacuse this is used for prediction
    
    # Removing information about location of the symbol on the canvas
    symb_mean = sum( [el[0]+el[1] for el in points] ) / len(points)   # Arithmetic mean value for the current symbol.
    new_p = [ (p_[0]-symb_mean, p_[1]-symb_mean) for p_ in points]
    
    # Scaling the symbol
    scaled_points = list()
    
    square_distance = lambda x,y: sum([(xi-yi)**2 for xi, yi in zip(x,y)])    
    max_square_dist = 0
    for p in combinations(new_p, 2):
        dist = square_distance(*p)
        if dist > max_square_dist:
            max_square_dist = dist
            max_pair = p
                
    mx = max( [abs(el[0]) for el in max_pair] )
    my = max( [abs(el[1]) for el in max_pair] )
    m = max(mx, my)

    scaled_points = [(el[0]/m, el[1]/m) for el in new_p]
    
    representative_points = list()
    D = 0
    for i in range(0, len(scaled_points)-1):
        D += square_distance(x=scaled_points[i], y=scaled_points[i+1])
        
    for k in range(M):
        dist_ = (k * D) / (M - 1)
        all_distances = [ abs( dist_ - square_distance(x=p_, y=scaled_points[0]) ) for p_ in scaled_points]  # Subtract the distance of each point from the start from the representative distance and the smallest value is the clossest
        closset_point = scaled_points[ np.argmin( np.array(all_distances) ) ]
        
        representative_points.append( closset_point )
    
    array = []
    for x in representative_points:
        array.append(x[0])
        array.append(x[1])
        
    array = np.array(array)
    
    return array


def preprocess_data(raw_dataset=""):

    classes = ["alpha", "beta", "gamma", "delta", "epsilon"]   

    M = 20

    with open("raw_dataset.pickle", "rb") as file:
        ds = pickle.load(file)
    
    points = [coord[0] for coord in ds]
    labels = [coord[1] for coord in ds]
    
    reverse_direction = True
    if reverse_direction:    # If I want to include all the symbols drawn the other way.
        extended_ds = list()
        for symb, lab in zip(points, labels):
            reverse_symb = list( reversed(symb) )    # Reverse the direction of the points of the current symbol
            # reverse_symb.append(lab)                    # Append the label of the symbol to the reversed direction.
            extended_ds.append( (symb, lab) )
            extended_ds.append( (reverse_symb, lab) )

        ds = deepcopy(extended_ds)
        points = [coord[0] for coord in ds]
        labels = [coord[1] for coord in ds]    

    # Removing information about location of the symbol on the canvas
    all_new_points = list()
    for p in points:
        symb_mean = sum( [el[0]+el[1] for el in p] ) / len(p)   # Arithmetic mean value for the current symbol.
        new_p = [ (p_[0]-symb_mean, p_[1]-symb_mean) for p_ in p]
        all_new_points.append(new_p)
    
    # Scaling the symbol
    scaled_points = list()
    points.clear()
    
    square_distance = lambda x,y: sum([(xi-yi)**2 for xi, yi in zip(x,y)])    
    for symbol_vec in all_new_points:
        max_square_dist = 0
        for p in combinations(symbol_vec, 2):
            dist = square_distance(*p)
            if dist > max_square_dist:
                max_square_dist = dist
                max_pair = p
                
        mx = max( [abs(el[0]) for el in max_pair] )
        my = max( [abs(el[1]) for el in max_pair] )
        m = max(mx, my)

        scaled_points.append( [(el[0]/m, el[1]/m) for el in symbol_vec] )
        
        
    # # Saving the scaled points to save some time while debugging.
    # with open("scaled_points.pickle", "wb") as file:
    #     pickle.dump(scaled_points, file)

    # exit(1)
    
    # with open("scaled_points.pickle", "rb") as file:
    #     scaled_points = pickle.load(file)
    
    prepared_dataset = list()
    representative_points = list()
    for symbol_vec, lab in zip(scaled_points, labels):
        D = 0
        for i in range(0, len(symbol_vec)-1):
            D += square_distance(x=symbol_vec[i], y=symbol_vec[i+1])
        
        for k in range(M):
            dist_ = (k * D) / (M - 1)
            all_distances = [ abs( dist_ - square_distance(x=p_, y=symbol_vec[0]) ) for p_ in symbol_vec]  # Subtract the distance of each point from the start from the representative distance and the smallest value is the clossest
            closset_point = symbol_vec[ np.argmin( np.array(all_distances) ) ]
            
            representative_points.append( closset_point )
            
        array = []
        for x in representative_points:
            array.append(x[0])
            array.append(x[1])
            
        array = np.array(array)
        
        oh_lab = np.zeros( len(classes) )
        oh_lab[classes.index(lab)] = 1.0    # One hot encoded vector
        prepared_dataset.append( (array, oh_lab) )
        representative_points.clear()
    
    with open("extended_prepared_dataset.pickle", "wb") as file:
        pickle.dump(prepared_dataset, file)

    return prepared_dataset

if __name__ == "__main__":
    preprocess_data()


    # # NOTE: WORKS!!!
    # from random import randint
    # vec = [(randint(0, 800), randint(0, 800)) for i in range(150)]
    # preprocess_symbol_vector(vec) 