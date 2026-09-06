# This file is for getting the Persistence Vector and the Betti Function for a given persistence diagram.

import numpy as np
import pandas as pd
import matplotlib as plt
import wfdb
import os
import os.path

from pathlib import Path
import csv
import time

import Visualize_EKG as vekg
import Normalize_EKG as nekg
import SWE as swe
import construct_complex as cc


def area_of_tri(b,d):

    height = d - b

    area = (height**2) / 2

    return area


def area_of_trap(d_1, b_2, d_2):

    # This is only for when the b_2 < d_1. If b_2 >= d_2, then use area_of_tri.
    # b_1 and d_1 are the birth and death times of the point that happens sooner in the time than b_2, d_2.
    # You don't need b_1 for this calculation.

    left_side = d_2 - d_1
    base = d_1 - b_2

    area = (left_side * base) + ((left_side**2) / 2)

    return area


"""
In all the following vectorization methods, the persistence diagram is assumed to be imprted as an np.array.
"""


def per_vec_dim(persistence_diagram, dimension): # This the the regular CPV where you do not consider the ovelapping triangles.

    count = 0
    total_area = 0

    i = 0
    while i < len(persistence_diagram):
        if persistence_diagram[i,0] == dimension and persistence_diagram[i,2] != float("inf"):
            b = persistence_diagram[i,1]
            d = persistence_diagram[i,2]
            triangle = area_of_tri(b,d)

            total_area = total_area + triangle
            count = count + 1
            i = i+1
        else:
            i = i+1
    
    return [count, total_area]


def per_vec_dim_noarea(persistence_diagram, dimension): # This is the CPV (no area) which is equivalent to the BV (N=1).

    count = 0

    i = 0
    while i < len(persistence_diagram):
        if persistence_diagram[i,0] == dimension and persistence_diagram[i,2] != float("inf"):
            count = count + 1
            i = i+1
        else:
            i = i+1
    
    return [count]


def per_vec_dim_nocount(persistence_diagram, dimension): # This is the CPV (no count).

    total_area = 0

    i = 0
    while i < len(persistence_diagram):
        if persistence_diagram[i,0] == dimension and persistence_diagram[i,2] != float("inf"):
            b = persistence_diagram[i,1]
            d = persistence_diagram[i,2]
            triangle = area_of_tri(b,d)

            total_area = total_area + triangle
            i = i+1
        else:
            i = i+1
    
    return [total_area]


def per_vec_dim_overlap(persistence_diagram, dimension): # This is the CPV (no overlap) where you do consider the overlapping triangles.

    count = 0
    total_area = 0

    if dimension == 0:
        i = 0
        max_death = 0
        while i< len(persistence_diagram):
            if persistence_diagram[i,2] > max_death and persistence_diagram[i,2] != float("inf"):
                max_death = persistence_diagram[i,2]
                count = count + 1
                i = i+1
            else:
                i = i+1
        
        total_area = area_of_tri(0,max_death)
        count = np.arctan(count)
    else:
        persistence_diagram = persistence_diagram[persistence_diagram[:, 1].argsort()] # to sort so that all the birth times are in order.

        i = 0
        indices_of_max_points = []
        while i < len(persistence_diagram):
            b = persistence_diagram[i,1]
            d = persistence_diagram[i,2]
            dim = persistence_diagram[i,0]

            if dim == dimension:
                if len(indices_of_max_points) == 0:
                    indices_of_max_points.append(i)
                    count = count+1
                    i = i+1
                else:
                    length = len(indices_of_max_points)
                    last_index = indices_of_max_points[length-1]

                    if b >= persistence_diagram[last_index,1] and d <= persistence_diagram[last_index,2] and d != float("inf") and dim == dimension:
                        count = count +1
                        i = i+1
                    elif b >= persistence_diagram[last_index,1] and d > persistence_diagram[last_index,2] and d != float("inf") and dim == dimension:
                        count = count +1
                        indices_of_max_points.append(i)
                        i = i+1
                    else:
                        i = i+1
            else:
                i = i+1
        
        i = 0
        while i < len(indices_of_max_points):
            current = indices_of_max_points[i]
            b = persistence_diagram[current,1]
            d = persistence_diagram[current,2]
            if i == 0:
                initial_area = area_of_tri(b,d)
                total_area = total_area + initial_area
                i = i+1
            else:
                previous = indices_of_max_points[i-1]
                d_1 = persistence_diagram[previous,2]
                if b >= d_1:
                    area = area_of_tri(b,d)
                    total_area = total_area + area
                    i = i+1
                else:
                    area = area_of_trap(d_1, b, d)
                    total_area = total_area + area
                    i = i+1
        
    return [count, total_area]


def per_vec_dim_nocount_overlap(persistence_diagram, dimension): # This is the CPV (No overlap and no count).

    total_area = 0

    if dimension == 0:
        i = 0
        max_death = 0
        while i< len(persistence_diagram):
            if persistence_diagram[i,2] > max_death and persistence_diagram[i,2] != float("inf"):
                max_death = persistence_diagram[i,2]
                i = i+1
            else:
                i = i+1
        
        total_area = area_of_tri(0,max_death)
    else:
        persistence_diagram = persistence_diagram[persistence_diagram[:, 1].argsort()] # to sort so that all the birth times are in order.

        i = 0
        indices_of_max_points = []
        while i < len(persistence_diagram):
            b = persistence_diagram[i,1]
            d = persistence_diagram[i,2]
            dim = persistence_diagram[i,0]

            if dim == dimension:
                if len(indices_of_max_points) == 0:
                    indices_of_max_points.append(i)
                    i = i+1
                else:
                    length = len(indices_of_max_points)
                    last_index = indices_of_max_points[length-1]

                    if b >= persistence_diagram[last_index,1] and d <= persistence_diagram[last_index,2] and d != float("inf") and dim == dimension:
                        i = i+1
                    elif b >= persistence_diagram[last_index,1] and d > persistence_diagram[last_index,2] and d != float("inf") and dim == dimension:
                        indices_of_max_points.append(i)
                        i = i+1
                    else:
                        i = i+1
            else:
                i = i+1
        
        i = 0
        while i < len(indices_of_max_points):
            current = indices_of_max_points[i]
            b = persistence_diagram[current,1]
            d = persistence_diagram[current,2]
            if i == 0:
                initial_area = area_of_tri(b,d)
                total_area = total_area + initial_area
                i = i+1
            else:
                previous = indices_of_max_points[i-1]
                d_1 = persistence_diagram[previous,2]
                if b >= d_1:
                    area = area_of_tri(b,d)
                    total_area = total_area + area
                    i = i+1
                else:
                    area = area_of_trap(d_1, b, d)
                    total_area = total_area + area
                    i = i+1
        
    return [total_area]


def betti_fun(persistence_diagram, dimension, pat_id, lead, N): # This is to get the discretized Betti Curve as a vector.

    i = 0
    bs = []
    ds = []
    while i<len(persistence_diagram):
        if persistence_diagram[i,2] == float("inf"):
            i = i+1
        else:
            bs.append(persistence_diagram[i,1])
            ds.append(persistence_diagram[i,2])
            i = i+1
    
    max_b = max(bs)
    max_d = max(ds)
    max_time = max(max_b, max_d)

    if N == 0:
        interval = max_time
    else:
        interval = max_time/N

    j = 0
    betti_vec = []
    betti_vec.append(pat_id)
    betti_vec.append(lead)
    while j < N+1:
        t = j*interval
        if dimension == 0:
            count = 0
            i = 0
            while i < len(persistence_diagram):
                if persistence_diagram[i,2] != float("inf") and persistence_diagram[i,0] == dimension and persistence_diagram[i,2] <= t+interval:
                    count = count + 1
                    i = i+1
                else:
                    i = i+1
            betti_vec.append(count)
        else:
            count = 0
            i = 0
            while i < len(persistence_diagram):
                if persistence_diagram[i,2] != float("inf") and persistence_diagram[i,0] == dimension:
                    if t <= persistence_diagram[i,1]  and persistence_diagram[i,1] < t+interval:
                        count = count+1
                        i = i+1
                    elif persistence_diagram[i,1]<t and persistence_diagram[i,2]>=t:
                        count = count+1
                        i = i+1
                    else:
                        i = i+1
                else:
                    i = i+1
            betti_vec.append(count)
        j = j+1

    if dimension == 0:
        j = len(betti_vec)
        last_val = betti_vec[j-1]
        output = []
        i = 2
        while i < len(betti_vec):
            output.append(betti_vec[i]) #You add 1 for the one connected component you have at the end of the persistent homology.
            i = i+1
        betti_vec = output
    else:
        j = len(betti_vec)
        output = []
        i = 2
        while i < len(betti_vec):
            output.append(betti_vec[i])
            i = i+1
        betti_vec = output


    return betti_vec

def put_together(thing_1, thing_2):
    
    output = []

    l1 = len(thing_1)
    l2 = len(thing_2)

    i = 0
    while i < l1+l2:
        if i < l1:
            output.append(thing_1[i])
            i = i+1
        elif i >= l1 and i < l1+l2:
            output.append(thing_2[i-l1])
            i = i+1
        else:
            i = i+1
    
    return output


"""
The following while loops perform the vectorizations for all the different vectorizations of interest. They use the above functions to find the area of the triangles formed
    by the features and the diagonal in each of the persistence diagrams.

The input of this file is the persistence diagrams as written in the CSV files from the main_EKG.py file.

The outputs of this file are the vectorizations of each of the persistence diagrams. These vectorizations are saved to a single CSV file based on the dimension of the SWE and the leads used.

"""


Rns = ['R2', 'R3', 'R4'] # If you have a specific R_n that you are interested in, specify that here.
Vectorizations = ['Betti_Vectorization', 'Pers_Vec_Vectorization', 'Pers_Vec_Vectorization_nooverlap', 'Pers_Vec_Vectorization_nocount', 'Pers_Vec_Vectorization_nocount_nooverlap', 'Pers_Vec_Vectorization_noarea']
ending_leads = ['_all', '_V1', '_V2', '_V3', '_V1_V2', '_V1_V3', '_V2_V3']

get_to_files = Path('./Brugada_dataset/files')
all_folder_names = []

for subdir in get_to_files.iterdir(): # makes a list of all the files names (which are the patient numbers)
    if subdir.is_dir():
        all_folder_names.append(subdir.name)

num_patients = len(all_folder_names)

annotation = np.genfromtxt('Brugada_dataset/metadata.csv', delimiter=',', skip_header=1)
annotation = np.array([annotation[:,0], annotation[:,3]]).T

ann = []
a = 0
while a < len(annotation):
    row = annotation[a]
    if row[1] == 0 or row[1] == 1:
        ann.append(row)
        a = a+1
    else:
        row = [row[0], 1]
        ann.append(row)
        a = a+1

ann = np.array(ann)

times = []

c = 0
while c < len(Vectorizations):
    vectorization = Vectorizations[c]

    r = 0
    while r < len(Rns):
        which_R = Rns[r]

        Ns = [0, 4, 9, 99] # For the Betti_Vector your input should be 1 less than you want actually want. These correspond to [1, 5, 10, 100] resp.

        el = 0
        while el < len(ending_leads):
            suffix = ending_leads[el]

            if vectorization == "Betti_Vectorization":
                n = 0
                while n < len(Ns):

                    start = time.time()

                    N = Ns[n]

                    output_csv_file = f"Vectorizations_9_1_26/Betti_Vectorization{suffix}_{N+1}_{which_R}.csv"
                        
                    which_one = output_csv_file.replace("Betti_Vectorization_","")
                    with open(output_csv_file, "w", newline="") as f:
                        writer = csv.writer(f)

                        if suffix == "_all":

                            i = 0
                            while i < num_patients:
                                pat_id = all_folder_names[i]

                                x = np.searchsorted(ann[:,0], float(pat_id))

                                value = ann[x,1]

                                lead_s = ['V1', 'V2', 'V3']

                                j = 0
                                while j < len(lead_s):
                                    lead = lead_s[j]
                                    pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                    dim = 0
                                    while dim < 2:
                                        if dim == 0:
                                            part_0 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                            dim = dim +1
                                        else:
                                            part_1 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                            dim = dim +1

                                    if j == 0:
                                        cpv = put_together(part_0, part_1)
                                        info = [pat_id]
                                        info_cpv = put_together(info, cpv)
                                    elif j == 1:
                                        cpv = put_together(part_0, part_1)
                                        info_cpv = put_together(info_cpv, cpv)
                                    elif j == 2:
                                        cpv = put_together(part_0, part_1)
                                        info_cpv = put_together(info_cpv, cpv)
                                        info_cpv.append(value)
                                        info_cpv = np.array(info_cpv)

                                    j = j+1

                                writer.writerow(info_cpv)

                                i = i+1

                        elif suffix == "_V1":

                            i = 0
                            while i < num_patients:
                                pat_id = all_folder_names[i]

                                x = np.searchsorted(ann[:,0], float(pat_id))

                                value = ann[x,1]

                                lead = "V1"
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                        dim = dim +1
                                    else:
                                        part_1 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                        dim = dim +1

                                cpv = put_together(part_0, part_1)
                                info = [pat_id]
                                info_cpv = put_together(info, cpv)
                                info_cpv.append(value)
                                info_cpv = np.array(info_cpv)

                                writer.writerow(info_cpv)

                                i = i+1

                        elif suffix == "_V2":

                            i = 0
                            while i < num_patients:
                                pat_id = all_folder_names[i]

                                x = np.searchsorted(ann[:,0], float(pat_id))

                                value = ann[x,1]

                                lead = "V2"
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                        dim = dim +1
                                    else:
                                        part_1 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                        dim = dim +1

                                cpv = put_together(part_0, part_1)
                                info = [pat_id]
                                info_cpv = put_together(info, cpv)
                                info_cpv.append(value)
                                info_cpv = np.array(info_cpv)

                                writer.writerow(info_cpv)

                                i = i+1

                        elif suffix == "_V3":

                            i = 0
                            while i < num_patients:
                                pat_id = all_folder_names[i]

                                x = np.searchsorted(ann[:,0], float(pat_id))

                                value = ann[x,1]

                                lead = "V3"
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                        dim = dim +1
                                    else:
                                        part_1 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                        dim = dim +1

                                cpv = put_together(part_0, part_1)
                                info = [pat_id]
                                info_cpv = put_together(info, cpv)
                                info_cpv.append(value)
                                info_cpv = np.array(info_cpv)

                                writer.writerow(info_cpv)

                                i = i+1

                        elif suffix == "_V1_V2":

                            i = 0
                            while i < num_patients:
                                pat_id = all_folder_names[i]

                                x = np.searchsorted(ann[:,0], float(pat_id))

                                value = ann[x,1]

                                lead_s = ["V1", "V2"]

                                j = 0
                                while j < len(lead_s):
                                    lead = lead_s[j]
                                    pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                    dim = 0
                                    while dim < 2:
                                        if dim == 0:
                                            part_0 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                            dim = dim +1
                                        else:
                                            part_1 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                            dim = dim +1

                                    if j == 0:
                                        cpv = put_together(part_0, part_1)
                                        info = [pat_id]
                                        info_cpv = put_together(info, cpv)
                                    elif j == 1:
                                        cpv = put_together(part_0, part_1)
                                        info_cpv = put_together(info_cpv, cpv)
                                        info_cpv.append(value)
                                        info_cpv = np.array(info_cpv)

                                    j = j+1

                                writer.writerow(info_cpv)

                                i = i+1

                        elif suffix == "_V1_V3":

                            i = 0
                            while i < num_patients:
                                pat_id = all_folder_names[i]

                                x = np.searchsorted(ann[:,0], float(pat_id))

                                value = ann[x,1]

                                lead_s = ["V1", "V3"]

                                j = 0
                                while j < len(lead_s):
                                    lead = lead_s[j]
                                    pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                    dim = 0
                                    while dim < 2:
                                        if dim == 0:
                                            part_0 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                            dim = dim +1
                                        else:
                                            part_1 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                            dim = dim +1

                                    if j == 0:
                                        cpv = put_together(part_0, part_1)
                                        info = [pat_id]
                                        info_cpv = put_together(info, cpv)
                                    elif j == 1:
                                        cpv = put_together(part_0, part_1)
                                        info_cpv = put_together(info_cpv, cpv)
                                        info_cpv.append(value)
                                        info_cpv = np.array(info_cpv)

                                    j = j+1

                                writer.writerow(info_cpv)

                                i = i+1

                        elif suffix == "_V2_V3":

                            i = 0
                            while i < num_patients:
                                pat_id = all_folder_names[i]

                                x = np.searchsorted(ann[:,0], float(pat_id))

                                value = ann[x,1]

                                lead_s = ["V2", "V3"]

                                j = 0
                                while j < len(lead_s):
                                    lead = lead_s[j]
                                    pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                    dim = 0
                                    while dim < 2:
                                        if dim == 0:
                                            part_0 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                            dim = dim +1
                                        else:
                                            part_1 = betti_fun(pers_diagram, dim, pat_id, lead, N)
                                            dim = dim +1

                                    if j == 0:
                                        cpv = put_together(part_0, part_1)
                                        info = [pat_id]
                                        info_cpv = put_together(info, cpv)
                                    elif j == 1:
                                        cpv = put_together(part_0, part_1)
                                        info_cpv = put_together(info_cpv, cpv)
                                        info_cpv.append(value)
                                        info_cpv = np.array(info_cpv)

                                    j = j+1

                                writer.writerow(info_cpv)

                                i = i+1
                    
                    time.sleep(1)
                    end = time.time()

                    print(f"Total runtime of the Betti Vectorization for {which_R} with N = {N+1} for {which_one} is {end - start} seconds")

                    times.append([vectorization, which_R, suffix, N, (end-start)])

                    n = n+1
    
            elif vectorization == "Pers_Vec_Vectorization":

                start = time.time()

                output_csv_file = f"Vectorizations_9_1_26/Pers_Vec_Vectorization{suffix}_{which_R}.csv"
                    
                which_one = output_csv_file.replace("Pers_Vec_Vectorization_","")
                with open(output_csv_file, "w", newline="") as f:
                    writer = csv.writer(f)

                    if suffix == "_all":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 0 area V1", "Total 1 Count V1", "Total 1 area V1", "Total 0 Count V2", "Total 0 area V2", "Total 1 Count V2", "Total 1 area V2", "Total 0 Count V3", "Total 0 area V3", "Total 1 Count V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ['V1', 'V2', 'V3']

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                elif j == 2:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 0 area V1", "Total 1 Count V1", "Total 1 area V1", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V1"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V2":

                        writer.writerow(["Patient_ID", "Total 0 Count V2", "Total 0 area V2", "Total 1 Count V2", "Total 1 area V2", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V2"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V3":

                        writer.writerow(["Patient_ID", "Total 0 Count V3", "Total 0 area V3", "Total 1 Count V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V3"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1_V2":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 0 area V1", "Total 1 Count V1", "Total 1 area V1", "Total 0 Count V2", "Total 0 area V2", "Total 1 Count V2", "Total 1 area V2", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V1", "V2"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1_V3":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 0 area V1", "Total 1 Count V1", "Total 1 area V1", "Total 0 Count V3", "Total 0 area V3", "Total 1 Count V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V1", "V3"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V2_V3":

                        writer.writerow(["Patient_ID", "Total 0 Count V2", "Total 0 area V2", "Total 1 Count V2", "Total 1 area V2", "Total 0 Count V3", "Total 0 area V3", "Total 1 Count V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V2", "V3"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1
                    
                    time.sleep(1)
                    end = time.time()

                    print(f"Total runtime of the Cumulative Persistence Vectorization for {which_R} for {which_one} is {end - start} seconds")

                    times.append([vectorization, which_R, suffix, 0, (end-start)])

            elif vectorization == "Pers_Vec_Vectorization_nooverlap":

                start = time.time()

                output_csv_file = f"Vectorizations_9_1_26/Pers_Vec_Vectorization_nooverlap{suffix}_{which_R}.csv"
                    
                which_one = output_csv_file.replace("Pers_Vec_Vectorization_nooverlap_","")
                with open(output_csv_file, "w", newline="") as f:
                    writer = csv.writer(f)

                    if suffix == "_all":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 0 area V1", "Total 1 Count V1", "Total 1 area V1", "Total 0 Count V2", "Total 0 area V2", "Total 1 Count V2", "Total 1 area V2", "Total 0 Count V3", "Total 0 area V3", "Total 1 Count V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ['V1', 'V2', 'V3']

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_overlap(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_overlap(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                elif j == 2:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 0 area V1", "Total 1 Count V1", "Total 1 area V1", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V1"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_overlap(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_overlap(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V2":

                        writer.writerow(["Patient_ID", "Total 0 Count V2", "Total 0 area V2", "Total 1 Count V2", "Total 1 area V2", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V2"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_overlap(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_overlap(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V3":

                        writer.writerow(["Patient_ID", "Total 0 Count V3", "Total 0 area V3", "Total 1 Count V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V3"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_overlap(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_overlap(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1_V2":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 0 area V1", "Total 1 Count V1", "Total 1 area V1", "Total 0 Count V2", "Total 0 area V2", "Total 1 Count V2", "Total 1 area V2", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V1", "V2"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_overlap(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_overlap(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1_V3":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 0 area V1", "Total 1 Count V1", "Total 1 area V1", "Total 0 Count V3", "Total 0 area V3", "Total 1 Count V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V1", "V3"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_overlap(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_overlap(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V2_V3":

                        writer.writerow(["Patient_ID", "Total 0 Count V2", "Total 0 area V2", "Total 1 Count V2", "Total 1 area V2", "Total 0 Count V3", "Total 0 area V3", "Total 1 Count V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V2", "V3"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_overlap(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_overlap(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1
                    
                    time.sleep(1)
                    end = time.time()

                    print(f"Total runtime of the Cumulative Persistence Vectorization no overlapping triangles for {which_R} for {which_one} is {end - start} seconds")

                    times.append([vectorization, which_R, suffix, 0, (end-start)])

            elif vectorization == "Pers_Vec_Vectorization_nocount":

                start = time.time()

                output_csv_file = f"Vectorizations_9_1_26/Pers_Vec_Vectorization_nocount{suffix}_{which_R}.csv"
                    
                which_one = output_csv_file.replace("Pers_Vec_Vectorization_nocount_","")
                with open(output_csv_file, "w", newline="") as f:
                    writer = csv.writer(f)

                    if suffix == "_all":

                        writer.writerow(["Patient_ID", "Total 0 area V1", "Total 1 area V1", "Total 0 area V2", "Total 1 area V2", "Total 0 area V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ['V1', 'V2', 'V3']

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_nocount(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_nocount(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                elif j == 2:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1":

                        writer.writerow(["Patient_ID", "Total 0 area V1", "Total 1 area V1", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V1"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_nocount(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_nocount(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V2":

                        writer.writerow(["Patient_ID", "Total 0 area V2", "Total 1 area V2", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V2"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_nocount(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_nocount(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V3":

                        writer.writerow(["Patient_ID", "Total 0 area V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V3"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_nocount(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_nocount(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1_V2":

                        writer.writerow(["Patient_ID", "Total 0 area V1", "Total 1 area V1", "Total 0 area V2", "Total 1 area V2", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V1", "V2"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_nocount(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_nocount(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1_V3":

                        writer.writerow(["Patient_ID", "Total 0 area V1", "Total 1 area V1", "Total 0 area V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V1", "V3"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_nocount(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_nocount(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V2_V3":

                        writer.writerow(["Patient_ID", "Total 0 area V2", "Total 1 area V2", "Total 0 area V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V2", "V3"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_nocount(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_nocount(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1
                    
                    time.sleep(1)
                    end = time.time()

                    print(f"Total runtime of the Cumulative Persistence Vectorization no count for {which_R} for {which_one} is {end - start} seconds")

                    times.append([vectorization, which_R, suffix, 0, (end-start)])

            elif vectorization == "Pers_Vec_Vectorization_nocount_nooverlap":

                start = time.time()

                output_csv_file = f"Vectorizations_9_1_26/Pers_Vec_Vectorization_nocount_nooverlap{suffix}_{which_R}.csv"
                    
                which_one = output_csv_file.replace("Pers_Vec_Vectorization_nocount_nooverlap_","")
                with open(output_csv_file, "w", newline="") as f:
                    writer = csv.writer(f)

                    if suffix == "_all":

                        writer.writerow(["Patient_ID", "Total 0 area V1", "Total 1 area V1", "Total 0 area V2", "Total 1 area V2", "Total 0 area V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ['V1', 'V2', 'V3']

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                elif j == 2:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1":

                        writer.writerow(["Patient_ID", "Total 0 area V1", "Total 1 area V1", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V1"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V2":

                        writer.writerow(["Patient_ID", "Total 0 area V2", "Total 1 area V2", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V2"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V3":

                        writer.writerow(["Patient_ID", "Total 0 area V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V3"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1_V2":

                        writer.writerow(["Patient_ID", "Total 0 area V1", "Total 1 area V1", "Total 0 area V2", "Total 1 area V2", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V1", "V2"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1_V3":

                        writer.writerow(["Patient_ID", "Total 0 area V1", "Total 1 area V1", "Total 0 area V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V1", "V3"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V2_V3":

                        writer.writerow(["Patient_ID", "Total 0 area V2", "Total 1 area V2", "Total 0 area V3", "Total 1 area V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V2", "V3"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1
                    
                    time.sleep(1)
                    end = time.time()

                    print(f"Total runtime of the Cumulative Persistence Vectorization no count and no overlapping triangles for {which_R} for {which_one} is {end - start} seconds")

                    times.append([vectorization, which_R, suffix, 0, (end-start)])
            
            elif vectorization == "Pers_Vec_Vectorization_noarea":

                start = time.time()

                output_csv_file = f"Vectorizations_9_1_26/Pers_Vec_Vectorization_noarea{suffix}_{which_R}.csv"
                    
                which_one = output_csv_file.replace("Pers_Vec_Vectorization_noarea_","")
                with open(output_csv_file, "w", newline="") as f:
                    writer = csv.writer(f)

                    if suffix == "_all":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 1 Count V1", "Total 0 Count V2", "Total 1 Count V2", "Total 0 Count V3", "Total 1 Count V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ['V1', 'V2', 'V3']

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_noarea(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_noarea(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                elif j == 2:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 1 Count V1", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V1"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_noarea(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_noarea(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V2":

                        writer.writerow(["Patient_ID", "Total 0 Count V2", "Total 1 Count V2", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V2"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_noarea(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_noarea(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V3":

                        writer.writerow(["Patient_ID", "Total 0 Count V3", "Total 1 Count V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead = "V3"
                            pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                            dim = 0
                            while dim < 2:
                                if dim == 0:
                                    part_0 = per_vec_dim_noarea(pers_diagram, dim)
                                    dim = dim +1
                                else:
                                    part_1 = per_vec_dim_noarea(pers_diagram, dim)
                                    dim = dim +1

                            cpv = put_together(part_0, part_1)
                            info = [pat_id]
                            info_cpv = put_together(info, cpv)
                            info_cpv.append(value)
                            info_cpv = np.array(info_cpv)

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1_V2":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 1 Count V1", "Total 0 Count V2", "Total 1 Count V2", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V1", "V2"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_noarea(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_noarea(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V1_V3":

                        writer.writerow(["Patient_ID", "Total 0 Count V1", "Total 1 Count V1", "Total 0 Count V3", "Total 1 Count V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V1", "V3"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_noarea(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_noarea(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1

                    elif suffix == "_V2_V3":

                        writer.writerow(["Patient_ID", "Total 0 Count V2", "Total 1 Count V2", "Total 0 Count V3", "Total 1 Count V3", "Value"])

                        i = 0
                        while i < num_patients:
                            pat_id = all_folder_names[i]

                            x = np.searchsorted(ann[:,0], float(pat_id))

                            value = ann[x,1]

                            lead_s = ["V2", "V3"]

                            j = 0
                            while j < len(lead_s):
                                lead = lead_s[j]
                                pers_diagram = np.genfromtxt(f'Persistent_Homology_{which_R}/{lead}/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                                dim = 0
                                while dim < 2:
                                    if dim == 0:
                                        part_0 = per_vec_dim_noarea(pers_diagram, dim)
                                        dim = dim +1
                                    else:
                                        part_1 = per_vec_dim_noarea(pers_diagram, dim)
                                        dim = dim +1

                                if j == 0:
                                    cpv = put_together(part_0, part_1)
                                    info = [pat_id]
                                    info_cpv = put_together(info, cpv)
                                elif j == 1:
                                    cpv = put_together(part_0, part_1)
                                    info_cpv = put_together(info_cpv, cpv)
                                    info_cpv.append(value)
                                    info_cpv = np.array(info_cpv)

                                j = j+1

                            writer.writerow(info_cpv)

                            i = i+1
                    
                    time.sleep(1)
                    end = time.time()

                    print(f"Total runtime of the Cumulative Persistence Vectorization no area for {which_R} for {which_one} is {end - start} seconds")

                    times.append([vectorization, which_R, suffix, 0, (end-start)])
        
            el = el+1

        r = r+1

    c = c+1


output_csv_file = "Time_Record_for_Vectorization.csv"
    
with open(output_csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Vectorization_Method", "Dimension", "Leads", "N", "TIME"])

    i = 0
    while i<len(times):
        writer.writerow(times[i])
        i = i+1

print("COMPLETED THE CODE....")
