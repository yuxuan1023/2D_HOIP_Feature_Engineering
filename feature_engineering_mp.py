#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ase
from ase.io import read
from ase.visualize import view
import numpy as np
from copy import copy, deepcopy
import scipy.spatial as ss
import os
from tqdm import tqdm
import pandas as pd
import itertools 
from multiprocessing import Pool


# In[2]:


def locate_point(pt, vec, dist):
    unit_vec = vec / np.linalg.norm(vec)
    return pt + dist * unit_vec


# In[3]:


def unit_vector(vector):
    return vector / np.linalg.norm(vector)
def get_angle(Pb1, Pb2, bridge_X_proj):
    v1=Pb1 - bridge_X_proj
    v2=Pb2 - bridge_X_proj
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees( np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) )


# In[4]:


def get_plane(P1, P2, Q):
    N = np.cross(P1,P2)
    d = np.dot(N, Q)
    return N[0], N[1], N[2], -d
def get_plane_dist(N, a,b,c,d):
    return abs(N[0]*a+N[1]*b+N[2]*c+d) / (a**2+b**2+c**2)**0.5


# In[5]:


def proj_point_plane(P_origin, P_plane, v_norm):
    a,b,c=v_norm[0],v_norm[1],v_norm[2]
    x,y,z=P_origin[0],P_origin[1],P_origin[2]
    d,e,f=P_plane[0],P_plane[1],P_plane[2]
    
    t=(a*d-a*x+b*e-b*y+c*f-c*z)/(a**2+b**2+c**2)
    return np.array([ x+t*a, y+t*b, z+t*c])


# In[6]:


def get_features(atoms, Pb_target):
    atoms_new=deepcopy(atoms)
    all_X_sym=[sym for sym in atoms.symbols if sym in ['I', 'Br', 'Cl']]
    X_symbol=max(set(all_X_sym), key=all_X_sym.count)
    Pbs=[i for i,atom in enumerate(atoms.symbols) if atom=='Pb']
    Pb_1=Pb_target
    Xs=[i for i,atom in enumerate(atoms.symbols) if atom==X_symbol]
    Xs_new=[]
    for X in Xs:
        for i in range(3):
            pending_X=locate_point(atoms.positions[X], -atoms.cell[i], np.linalg.norm(atoms.cell[i]))
            if np.all([np.linalg.norm(pending_X - atoms.positions[X_])>1e-3 for X_ in Xs] ) : 
                Xs_new.append( pending_X )
            pending_X=locate_point(atoms.positions[X], atoms.cell[i], np.linalg.norm(atoms.cell[i]))
            if np.all(np.linalg.norm(pending_X - atoms.positions[X_])>1e-3 for X_ in Xs )  :
                Xs_new.append( pending_X )
    Xs_new2=[]
    for X in Xs_new:
        for i in range(3):
            pending_X=locate_point(X, -atoms.cell[i], np.linalg.norm(atoms.cell[i]))
            if np.all([np.linalg.norm( pending_X -atoms.positions[X_])>1e-3 for X_ in Xs]) and                         np.all([np.linalg.norm( pending_X - X__)>1e-3 for X__ in Xs_new]):
                Xs_new2.append( pending_X )
            pending_X=locate_point(X, atoms.cell[i], np.linalg.norm(atoms.cell[i]))
            if np.all([np.linalg.norm( pending_X -atoms.positions[X_])>1e-3 for X_ in Xs]) and                         np.all([np.linalg.norm( pending_X - X__)>1e-3 for X__ in Xs_new]):
                Xs_new2.append( pending_X )
    Xs_new.extend(Xs_new2)
    Xs_new3=[]
    for X in Xs_new2:
        for i in range(3):
            pending_X=locate_point(X, -atoms.cell[i], np.linalg.norm(atoms.cell[i]))
            if np.all([np.linalg.norm( pending_X -atoms.positions[X_])>1e-3 for X_ in Xs]) and                         np.all([np.linalg.norm( pending_X - X__)>1e-3 for X__ in Xs_new]):
                Xs_new3.append( pending_X )
            pending_X=locate_point(X, atoms.cell[i], np.linalg.norm(atoms.cell[i]))
            if np.all([np.linalg.norm( pending_X -atoms.positions[X_])>1e-3 for X_ in Xs]) and                         np.all([np.linalg.norm( pending_X - X__)>1e-3 for X__ in Xs_new]):
                Xs_new3.append( pending_X )
    Xs_new.extend(Xs_new3)
    
    for X in Xs_new:
        atoms.append(X_symbol)
        atoms.positions[-1] = X
    
    Xs_all=[i for i,atom in enumerate(atoms.symbols) if atom==X_symbol]
    Xs_all_kept=[]
    for i,X in enumerate(Xs_all):
        if np.all([np.linalg.norm(atoms.positions[X]-atoms.positions[X_])>1e-3 for X_ in Xs_all[:i]]):
            Xs_all_kept.append(X)
    Pb1_X_dists=[ np.linalg.norm( atoms.positions[X]-atoms.positions[Pb_1] ) for X in Xs_all_kept]
    Xs_1=[ Xs_all_kept[i] for i in np.array(Pb1_X_dists).argsort()[:6] ]

    #Pb2_X_dists=[ np.linalg.norm( atoms.positions[X]-atoms.positions[Pb_2] ) for X in Xs_1]
    #bridge_X = Xs_1[np.array(Pb2_X_dists).argsort()[0]]
    Pbs_new=[]
    for Pb in Pbs:
        for i in range(3):
            Pbs_new.append( locate_point(atoms.positions[Pb], -atoms.cell[i], np.linalg.norm(atoms.cell[i])))
            Pbs_new.append( locate_point(atoms.positions[Pb], atoms.cell[i], np.linalg.norm(atoms.cell[i])))
    Pbs_new2=[]
    for Pb in Pbs:
        for i in range(3):
            Pbs_new2.append( locate_point(atoms.positions[Pb], -atoms.cell[i], np.linalg.norm(atoms.cell[i])))
            Pbs_new2.append( locate_point(atoms.positions[Pb], atoms.cell[i], np.linalg.norm(atoms.cell[i])))
    Pbs_new.extend(Pbs_new2)
    for Pb in Pbs_new:
        atoms.append('Pb')
        atoms.positions[-1] = Pb
    
    Pbs_all=[i for i,atom in enumerate(atoms.symbols) if atom=='Pb']
    Pbs_all_kept=[]
    for i,Pb in enumerate(Pbs_all):
        if Pb != Pb_1 and np.all([np.linalg.norm(atoms.positions[Pb]-atoms.positions[Pb_])>1e-3 for Pb_ in Pbs_all[:i]]):
            Pbs_all_kept.append(Pb)

    Pb1_Pb_dists=[ np.linalg.norm( atoms.positions[Pb]-atoms.positions[Pb_1] ) for Pb in Pbs_all_kept]
    Pb_bridge=[Pbs_all_kept[i] for i in np.array(Pb1_Pb_dists).argsort()[:1]][0] #the nearest Pb with Pb1
    if Pb_bridge not in Pbs:
        atoms_new.append('Pb')
        atoms_new.positions[-1] = atoms.positions[Pb_bridge]
    Pb_bridge_dists=[ np.linalg.norm( atoms.positions[X]-atoms.positions[Pb_bridge] ) for X in Xs_1]
    bridge_X = Xs_1[np.array(Pb_bridge_dists).argsort()[0]]
    
    X_added=[X for X in Xs_1 if X not in Xs]
    if len(X_added)>0:
        for i,X in enumerate(X_added):
            atoms_new.append(X_symbol)
            atoms_new.positions[-1] = atoms.positions[X]
    feature1=atoms.get_angle( Pb_1, bridge_X, Pb_bridge )
    print(1,feature1)
    Ns=[i for i,atom in enumerate(atoms.symbols) if atom=='N']
    Ns_new=[]
    for N in Ns:
        for i in range(3):
            Ns_new.append( locate_point(atoms.positions[N], -atoms.cell[i], np.linalg.norm(atoms.cell[i])))
            Ns_new.append( locate_point(atoms.positions[N], atoms.cell[i], np.linalg.norm(atoms.cell[i])))
    Ns_new2=[]
    for N in Ns_new:
        for i in range(3):
            Ns_new2.append( locate_point(N, -atoms.cell[i], np.linalg.norm(atoms.cell[i])))
            Ns_new2.append( locate_point(N, atoms.cell[i], np.linalg.norm(atoms.cell[i])))
    Ns_new.extend(Ns_new2)
    Ns_new3=[]
    for N in Ns_new:
        for i in range(3):
            Ns_new3.append( locate_point(N, -atoms.cell[i], np.linalg.norm(atoms.cell[i])))
            Ns_new3.append( locate_point(N, atoms.cell[i], np.linalg.norm(atoms.cell[i])))
    Ns_new.extend(Ns_new3)
    
    for N in Ns_new:
        atoms.append('N')
        atoms.positions[-1] = N

    Ns_all=[i for i,atom in enumerate(atoms.symbols) if atom=='N']
    Ns_all_kept=[]
    for i,N in enumerate(Ns_all):
        if np.all([np.linalg.norm(atoms.positions[N]-atoms.positions[N_])>1e-3 for N_ in Ns_all[:i]]):
            Ns_all_kept.append(N)
    Pb1_N_dists=[ np.linalg.norm( atoms.positions[N]-atoms.positions[Pb_1] ) for N in Ns_all_kept]
    N_1=[Ns_all_kept[i] for i in np.array(Pb1_N_dists).argsort()[:1]][0] #the nearest N with Pb1
    if N_1 not in Ns:
        atoms_new.append('N')
        atoms_new.positions[-1] = atoms.positions[N_1]
    
    longest_cell=0
    same_cells=[]
    ans=[]
    plane_vs=[]
    for i in range(3):
        if i==2: j=0
        else: j=i+1
        if abs(abs(atoms.cell[i][0])-abs(atoms.cell[j][0]))<1e-3 and                 np.linalg.norm(atoms.cell[i])==np.linalg.norm(atoms.cell[j]):
            same_cells.extend([i,j])
            
    if len(same_cells)>0:
        longest_cell=[i for i in range(3) if i not in same_cells][0]
        plane_vs.append( unit_vector (atoms.cell[longest_cell]) )
        if np.degrees(np.arccos(np.dot( unit_vector(atoms.cell[0]), unit_vector(atoms.cell[1])))) >= 90:
            plane_vs.append( unit_vector( unit_vector(atoms.cell[0])+unit_vector(atoms.cell[1]) ) )
        else:
            plane_vs.append( unit_vector( unit_vector(atoms.cell[0])-unit_vector(atoms.cell[1]) ) )
        a,b,c,d=get_plane(plane_vs[0],plane_vs[1],atoms.positions[Pb_1])
        X_origin=Xs_1[ np.array([get_plane_dist(atoms.positions[X], a,b,c,d) for X in Xs_1]).argsort()[-1] ]
    else:
        Cs=[i for i,ele in enumerate(atoms.symbols) if ele=='C']
        r_all=[]
        for i in range(3):
            cell_vs=[]
            cell_vs=[atoms.cell[j] for j in range(3) if j!=i]
            a,b,c,d=get_plane(cell_vs[0], cell_vs[1], atoms.positions[Pb_1])
            Pb_1_new=locate_point(atoms.positions[Pb_1], atoms.cell[i], np.linalg.norm(atoms.cell[i]))
            r1=get_plane_dist(atoms.positions[Pb_1],a,b,c,d)
            r2=get_plane_dist(Pb_1_new, a,b,c,d)
            r_all.append(sum( [get_plane_dist(atoms.positions[C], a,b,c,d) for C in Cs] ) )
        longest_cell=np.array(r_all).argsort()[-1]
        for X in Xs_1:
            ans.append( np.degrees( np.arccos( np.dot( unit_vector(atoms.positions[X]-atoms.positions[Pb_1]),                                      unit_vector(atoms.cell[longest_cell] ) )) ) )
        X_origin=Xs_1[ np.array(ans).argsort()[0]]
        for i in range(3):
            if i!=longest_cell:
                plane_vs.append(atoms.cell[i])
        a,b,c,d=get_plane(plane_vs[0],plane_vs[1],atoms.positions[Pb_1])
    feature2=get_plane_dist(atoms.positions[N_1], a,b,c,d)
    print(2,feature2)
    bridge_X_proj=proj_point_plane( atoms.positions[bridge_X] ,atoms.positions[Pb_1], [a,b,c]/np.linalg.norm([a,b,c]))
    feature3=get_angle(atoms.positions[Pb_1], atoms.positions[Pb_bridge], bridge_X_proj)

    print(3,feature3)
    X_X_dists=[np.linalg.norm( atoms.positions[X_origin]-atoms.positions[X] ) for X in Xs_1 ]
    X_oppo=[Xs_1[i] for i in np.array(X_X_dists).argsort()[-1:]][0]
    Xs_around=[X for X in Xs_1 if X != X_origin and X != X_oppo]
    X_Pb_X=[]
    #X_origin
    for X in Xs_1:
        if X != X_origin and X != X_oppo:
            X_Pb_X.append(atoms.get_angle( X_origin, Pb_1, X ))
    #X_oopo
    for X in Xs_1:
        if X != X_origin and X != X_oppo:
            X_Pb_X.append(atoms.get_angle( X_oppo, Pb_1, X ))
    #Xs_around
    pairs=[]
    #X_Pb_X3=[]
    for i in Xs_around:
        angles=[]
        pair=[]
        for j in Xs_around:
            if i!=j and set([i,j]) not in pairs:
                angles.append( atoms.get_angle( i, Pb_1, j ) )
                pair.append( set([i,j]) )
        if len(angles) > 0:
            X_Pb_X.extend( [ angles[an] for an in np.array(angles).argsort()[:-1]] )
            pairs.extend( [pair[an] for an in np.array(angles).argsort()[:-1]] )
    feature4=np.mean(X_Pb_X[:8])
    feature5=np.mean(X_Pb_X[-4:])
    feature6=sum((angle-90)**2 for angle in X_Pb_X)/11
    print(4,feature4)
    print(5,feature5)
    print(6,feature6)
    X_X_lens_ax=[]
    for i in [X_origin, X_oppo]:
        for j in Xs_around:
            X_X_lens_ax.append(np.linalg.norm( atoms.positions[i]- atoms.positions[j] ) )
    feature7=np.mean(X_X_lens_ax)
    X_X_lens_eq=[]
    for pair in pairs:
        p=list(pair)
        X_X_lens_eq.append(np.linalg.norm( atoms.positions[p[0]]- atoms.positions[p[1]] ) )
    feature8=np.mean(X_X_lens_eq)
    #ax-bong-length
    feature9=np.mean([np.linalg.norm( atoms.positions[Pb_1] - atoms.positions[X]) for X in [X_origin, X_oppo] ])
    #eq-bong-length
    feature10=np.mean([np.linalg.norm( atoms.positions[Pb_1] - atoms.positions[X]) for X in Xs_around ])
    
    print(7,feature7)
    print(8,feature8)
    print(9,feature9)

    points=[atoms.positions[i] for i in Xs_1]
    hull = ss.ConvexHull(points)
    #print('volume inside points is: ',hull.volume)
    l_edge=( hull.volume*3/(2**0.5) )**(1./3.)
    d0=l_edge/(2**0.5)
    feature11=sum( [(np.linalg.norm( atoms.positions[Pb_1] - atoms.positions[X] )/d0)**2 for X in Xs_1] )/6
    feature12=np.mean([np.degrees(np.arccos(np.dot(        unit_vector(atoms.positions[X_origin]-atoms.positions[Pb_1]),unit_vector([a,b,c])))),                        np.degrees(np.arccos(np.dot( unit_vector(atoms.positions[X_oppo]-atoms.positions[Pb_1]),                                                    -unit_vector([a,b,c])))) ])
    if feature12>90:
        feature12=180-feature12
    features=[feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,              feature10,feature11,feature12]
    print(10,feature10)
    print(11,feature11)
    print(12,feature12)
    return atoms_new, [X_symbol]+features


# In[7]:


#get_features(read('290.vasp'))


# In[8]:


cwd=os.getcwd()


# In[ ]:


os.chdir(os.path.join(cwd, 'opt_outcars') )
filenames=filenames=os.listdir(os.path.join(cwd, 'opt_outcars'))
#filenames=filenames=os.listdir(os.getcwd())
data=[]
covs_all=[]
for filename in tqdm(filenames):
    if filename.endswith('.vasp'):
        #print(filename)
        atoms = read(filename) 
        Pbs = [i for i,atom in enumerate(atoms.symbols) if atom=='Pb']
        features_atom=[]
        print(filename, 'total',len(Pbs))
        #for Pb in tqdm(Pbs):
        #for j, Pb in enumerate(Pbs):
        #    print(filename, j)
        #    _, features=get_features( atoms, Pb )
        #    features_atom.append(features[1:])
        pool = Pool(processes = 4)
        features_allpb = pool.starmap( get_features , zip(itertools.cycle([atoms]), Pbs) )

        for f in features_allpb:
            #print('f',f)
            features_atom.append(f[1][1:])
            #print('?',features_atom )
        #for Pb in Pbs:
        #_, features=get_features( read(filename), Pbs[0] )
        #features_atom.append(features[1:])
        #X=features[0]
        X=f[1][0]
        data.append( [ filename.split('.')[0] ] + [X] + list( np.mean(features_atom, axis=0) ) )


# In[ ]:


#os.getcwd()


# In[ ]:


df=pd.DataFrame(data, columns=['idx','X','Pb_X_Pb', 'N_penetration', 'in_plane_Pb_X_Pb','ax_bond_angle',                            'eq_bond_angle','bond_angle_variance','ax_X_X','eq_X_X','ax_bond_length',                            'eq_bond_length','elongation','out_of_plane_disortion'])


# In[ ]:


#os.chdir('D:\\myhair\\0_perovskite\\0717trasnfer_learning')


# In[ ]:

os.chdir(cwd)
#os.chdir('D:\\myhair\\2_proj_perovskite\\0911_transfer')
df.to_csv('features_tl_avg.csv', index=False)


# In[ ]:


df


# In[ ]:




