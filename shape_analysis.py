'''
This code was adapted from the supplementary data files of 
Fukunaga & Burns (2020) Metrics of Coral Reef Structural Complexity Extracted
from 3D Mesh Models and Digital Elevation Models. Remote Sensing 12, 2676. https://doi.org/10.3390/rs12172676
The article is available under the CC-BY 4.0 license.
The license of this code is GPL v.3
'''

import numpy as np
import math

from scipy import spatial
from scipy.stats import linregress

from sklearn.linear_model import LinearRegression

from multiprocessing import Pool
from functools import partial


### File importing
def obj_importer(filepath):
    '''
    Imports a wavefront obj file and returns two objects that
    hold information about the vertices and faces respectively.
    The files should be conform to the most basic obj standard.
    The files must have faces coded with the vertex indices only.
    Faces must not be coded like "f 378//837 393//837 464//837"
    The correct format is "f 15351 15206 15536"
    '''
    v = []
    f = []

    file = open(filepath)
    
    for line in file:
        # Retrieving vertex coordinates
        if line[:2] == 'v ':
            index1 = line.find(' ') + 1
            index2 = line.find(' ', index1 + 1)
            index3 = line.find(' ', index2 + 1)
            
            vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
            v.append(vertex)

        # Retrieving face vertex coordinates
        elif line[0] == 'f':
            i = line.find(' ') + 1
            face = []
            for item in range(line.count(' ')):
                if line.find(' ', i) == -1:
                    face.append(line[i:-1])
                    break
                face.append(line[i:line.find(' ', i)])
                i = line.find(' ', i) + 1
            f.append(tuple(face))
    file.close()
    return(v, f)



### Vector dispersion
def calc_vector_dispersion(v,f):
    '''
    Reads faces of a mesh and calculates the vector dispersion (dispersion of normal vectors)
    '''
    # calcuate vector dispersion
    # if the number of vertices is V, vertex in [f] are numbered from 1 to V. Python will number each element of [v] from 0 to V-1.
    # adjust for this when obtaining the coodinates of vertices creating each face (pt1, pt2, pt3).
    sumCosx = 0
    sumCosy = 0
    sumCosz = 0
    skip = 0

    for i in range(0, len(f)):
        
        # get the 3 vertices for each face
        pt1 = v[int(f[i][0]) - 1]
        pt2 = v[int(f[i][1]) - 1]
        pt3 = v[int(f[i][2]) - 1]
        
        # define vec2 (pt2 - pt1) and vec3 (pt3 - pt1)
        vec2 = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1], pt2[2] - pt1[2]])
        vec3 = np.array([pt3[0] - pt1[0], pt3[1] - pt1[1], pt3[2] - pt1[2]])

        # also check vec1 (pt2 - pt3)
        vec1 = np.array([pt2[0] - pt3[0], pt2[1] - pt3[1], pt2[2] - pt3[2]])

        # if vec1, vec2 or vec3 is [0, 0, 0], two of the three vertices have pretty much the same coordinates, so skipt that face
        if (vec1[0] == 0 and vec1[1] == 0 and vec1[2] == 0) or (vec2[0] == 0 and vec2[1] == 0 and vec2[2] == 0) or (vec3[0] == 0 and vec3[1] == 0 and vec3[2] == 0):
            skip += 1
            continue
        
        # get normal of each mesh surface
        vecNormal = np.cross(vec2, vec3)
        
        # get direction cosine of each normal vector
        cosDenom = math.sqrt(vecNormal[0] * vecNormal[0] + vecNormal[1] * vecNormal[1] + vecNormal[2] * vecNormal[2])
        cosNormal = np.array([vecNormal[0] / cosDenom, 
                            vecNormal[1] / cosDenom, 
                            vecNormal[2] / cosDenom])
        
        sumCosx += cosNormal[0]
        sumCosy += cosNormal[1]
        sumCosz += cosNormal[2]

    sumCosx2 = math.pow(sumCosx, 2)
    sumCosy2 = math.pow(sumCosy, 2)
    sumCosz2 = math.pow(sumCosz, 2)
    r1 = math.sqrt(sumCosx2 + sumCosy2 + sumCosz2)
    vk = round((len(f) - skip - r1) / (len(f) - skip - 1), 6)

    return(vk, skip)


# def do_stuff(point, v, r_eval):
#     vtree = spatial.cKDTree(v)
#     occurrence = 0
#     neighbor = vtree.query(point, k = 1, p = 2)
#     x_eval = (v[neighbor[1]][0] >= point[0] - (r_eval / 2)) and (v[neighbor[1]][0] < point[0] + (r_eval / 2))
#     y_eval = (v[neighbor[1]][1] >= point[1] - (r_eval / 2)) and (v[neighbor[1]][1] < point[1] + (r_eval / 2))
#     z_eval = (v[neighbor[1]][2] >= point[2] - (r_eval / 2)) and (v[neighbor[1]][2] < point[2] + (r_eval / 2))
#     if x_eval and y_eval and z_eval:
#         occurrence = 1
#     return(occurrence)


def query_tree(vtree, v, r_eval, point):
    neighbor = vtree.query(point, k = 1, p = 2)
    x_eval = (v[neighbor[1]][0] >= point[0] - (r_eval / 2)) and (v[neighbor[1]][0] < point[0] + (r_eval / 2))
    y_eval = (v[neighbor[1]][1] >= point[1] - (r_eval / 2)) and (v[neighbor[1]][1] < point[1] + (r_eval / 2))
    z_eval = (v[neighbor[1]][2] >= point[2] - (r_eval / 2)) and (v[neighbor[1]][2] < point[2] + (r_eval / 2))
    if x_eval and y_eval and z_eval:
        return 1
    else:
        return 0



def fd_cube(v, min_res):

    # kd-tree for quick nearest-neighbor lookup
    vtree = spatial.cKDTree(v)

    # convert vertices to numpy array
    vers = np.array(v)


    # determine the ranges of x, y and z with all vertices inclusive, r = max range of the x, y, z
    xmin = math.floor(min(vers[:, 0]) * 100) / 100
    xmax = math.ceil(max(vers[:, 0]) * 100) / 100
    ymin = math.floor(min(vers[:, 1]) * 100) / 100
    ymax = math.ceil(max(vers[:, 1]) * 100) / 100
    zmin = math.floor(min(vers[:, 2]) * 100) / 100
    zmax = math.ceil(max(vers[:, 2]) * 100) / 100
    r = round(max(xmax - xmin, ymax - ymin, zmax - zmin), 2)

    print('x_range = (', str(xmin), ', ', str(xmax), '), ',
      'y_range = (', str(ymin), ', ', str(ymax), '), ',
      'z_range = (', str(zmin), ', ', str(zmax), '), ',
      'r =', str(r))


    # obtain the mid point of x, y, z
    x_m = round((xmax + xmin) / 2, 3)
    y_m = round((ymax + ymin) / 2, 3)
    z_m = round((zmax + zmin) / 2, 3)
    print('x_mid =', str(x_m), ', ',
        'y_mid =', str(y_m), ', ',
        'z_mid =', str(z_m))
    

    # define the low and high bounds of x, y, z for cube counting and confirm all vertices are inside
    x_l = round(x_m - r / 2, 3)
    x_h = round(x_m + r / 2, 3)
    y_l = round(y_m - r / 2, 3)
    y_h = round(y_m + r / 2, 3)
    z_l = round(z_m - r / 2, 3)
    z_h = round(z_m + r / 2, 3)
    print('x_bound = (', str(x_l), ', ', str(x_h), '), ',
        'y_bound = (', str(y_l), ', ', str(y_h), '), ',
        'z_bound = (', str(z_l), ', ', str(z_h), ')')
    

    # set the sizes of cube for cube counting in the r_seq list
    r_seq = []
    r_test = r
    while r_test >= min_res:
        r_seq.append(r_test)
        r_test = r_test / 2
    print(r_seq)

    # initialize the count array to save the number of cubes containing vertices
    count = []
    for i in range(len(r_seq)):
        count.append(0)
    count[0] = 1   # for r, the number of cube required to contain all vertices is 1
    print(count)


    # get the center coordinates of cubes (of size r_eval) by setting 
    # the low to the low bound + half cube size and the high to the hi bound - half cube size, 
    # and use np.mgrid to get the sequences from the low to high using the cube size as the interval.
    # cKDTree.query queries the kd-tree for nearest neighbors
    # if the nearest neighbor is inside the cube (center coordinate - r_eval/2, center coordinate + r_eval/2),
    # the cube contains at least one vertex, so increase the count by 1.

    for i in range(1, len(r_seq)):
        r_eval = r_seq[i]
        x_low = x_l + (r_eval / 2)
        x_hi = x_h - (r_eval / 2) + 0.0001
        y_low = y_l + (r_eval / 2)
        y_hi = y_h - (r_eval / 2) + 0.0001
        z_low = z_l + (r_eval / 2)
        z_hi = z_h - (r_eval / 2) + 0.0001

        x, y, z = np.mgrid[x_low:x_hi:r_eval, y_low:y_hi:r_eval, z_low:z_hi:r_eval]
        points = zip(x.ravel(), y.ravel(), z.ravel())
        

        # for point in points:
        #     neighbor = vtree.query(point, k = 1, p = 2)
        #     x_eval = (v[neighbor[1]][0] >= point[0] - (r_eval / 2)) and (v[neighbor[1]][0] < point[0] + (r_eval / 2))
        #     y_eval = (v[neighbor[1]][1] >= point[1] - (r_eval / 2)) and (v[neighbor[1]][1] < point[1] + (r_eval / 2))
        #     z_eval = (v[neighbor[1]][2] >= point[2] - (r_eval / 2)) and (v[neighbor[1]][2] < point[2] + (r_eval / 2))
        #     if x_eval and y_eval and z_eval:
        #         count[i] += 1
        with Pool() as p:
            count[i] = sum(p.map(partial(query_tree, vtree, v, r_eval), points))

    
    # get the slope of r_seq and count in log scale, slope here is fractal dimension
    x = np.log(np.reciprocal(r_seq))
    y = np.log(count)
    r = linregress(x, y)
    x_0 = x - x[0]
    r_zero_intercept = LinearRegression(fit_intercept = False).fit(x_0.reshape(-1, 1), y.reshape(-1, 1))
    print(r_seq)
    print(count)
    print('fractal dimension =', str(r.slope))
    print('fractal dimension zero intercept =', str(r_zero_intercept.coef_))
    return(r.slope, r_zero_intercept.coef_)


class Shape:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = filepath.split("/")[-1]
        self.vertices, self.faces = obj_importer(filepath)
        self.vertex_count = len(self.vertices)
        self.face_count = len(self.faces)
        

    def vector_dispersion(self):
        vector_dispersion, skipped_faces = calc_vector_dispersion(self.vertices, self.faces)
        print('# of skipped faces =', str(skipped_faces))
        self.vector_dispersion = vector_dispersion
        return vector_dispersion
    
    def fractal_dim_cube_counting(self, min_res = 0.01):
        self.cube_counting_fractal_dimension, self.cube_counting_zero_intercept = fd_cube(self.vertices, min_res)
        return fd_cube(self.vertices, min_res)
    


synth_high_convexity = Shape("synthetic_shapes/high_convexity.obj")
synth_high_convexity.vector_dispersion()
synth_high_convexity.fractal_dim_cube_counting()

synth_low_convexity = Shape("synthetic_shapes/low_convexity.obj")
synth_low_convexity.vector_dispersion()
synth_low_convexity.fractal_dim_cube_counting()


synth_high_convexity.cube_counting_fractal_dimension
synth_low_convexity.cube_counting_fractal_dimension

synth_high_convexity.vector_dispersion
synth_low_convexity.vector_dispersion