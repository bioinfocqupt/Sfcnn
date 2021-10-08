#!/usr/bin/env python
# coding: utf-8


import os
from glob import glob
from openbabel import pybel
import numpy as np
import random
import pickle


#Converts the protein-ligand complexes into 4D tensor. 
class Feature_extractor():
    def __init__(self):
        self.atom_codes = {}
        #'others' includs metal atoms and B atom. There are no B atoms on training and test sets. 
        others = ([3,4,5,11,12,13]+list(range(19,32))+list(range(37,51))+list(range(55,84)))
        #C and N atoms can be hybridized in three ways and S atom can be hybridized in two ways here. 
        #Hydrogen atom is also considered for feature extraction.
        atom_types = [1,(6,1),(6,2),(6,3),(7,1),(7,2),(7,3),8,15,(16,2),(16,3),34,[9,17,35,53],others]
      
        for i, j in enumerate(atom_types):
            if type(j) is list:
                for k in j:
                    self.atom_codes[k] = i
                
            else:
                self.atom_codes[j] = i              
        
        self.sum_atom_types = len(atom_types)
        
    #Onehot encoding of each atom. The atoms in protein or ligand are treated separately.
    def encode(self, atomic_num, molprotein):
        encoding = np.zeros(self.sum_atom_types*2)
        if molprotein == 1:
            encoding[self.atom_codes[atomic_num]] = 1.0
        else:
            encoding[self.sum_atom_types+self.atom_codes[atomic_num]] = 1.0
        
        return encoding
    
    #Get atom coords and atom features from the complexes.   
    def get_features(self, molecule, molprotein):
        coords = []
        features = []
            
        for atom in molecule:
            coords.append(atom.coords)
            if atom.atomicnum in [6,7,16]:
                atomicnum = (atom.atomicnum,atom.hyb)
                features.append(self.encode(atomicnum,molprotein))
            else:
                features.append(self.encode(atom.atomicnum,molprotein))
        
        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        
        return coords, features
     
    #Define the rotation matrixs of 3D stuctures.
    def rotation_matrix(self, t, roller):
        if roller==0:
            return np.array([[1,0,0],[0,np.cos(t),np.sin(t)],[0,-np.sin(t),np.cos(t)]])
        elif roller==1:
            return np.array([[np.cos(t),0,-np.sin(t)],[0,1,0],[np.sin(t),0,np.cos(t)]])
        elif roller==2:
            return np.array([[np.cos(t),np.sin(t),0],[-np.sin(t),np.cos(t),0],[0,0,1]])

    #Generate 3d grid or 4d tensor. Each grid represents a voxel. Each voxel represents the atom in it by onehot encoding of atomic type.
    #Each complex in train set is rotated 9 times for data amplification.
    #The complexes in core set are not rotated. 
    #The default resolution is 20*20*20.
    def grid(self,coords, features, resolution=1.0, max_dist=10.0, rotations=9):
        assert coords.shape[1] == 3
        assert coords.shape[0] == features.shape[0]  

        
        grid=np.zeros((rotations+1,20,20,20,features.shape[1]),dtype=np.float32)
        x=y=z=np.array(range(-10,10),dtype=np.float32)+0.5
        for i in range(len(coords)):
            coord=coords[i]
            tmpx=abs(coord[0]-x)
            tmpy=abs(coord[1]-y)
            tmpz=abs(coord[2]-z)
            if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                grid[0,np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]
        
        for j in range(rotations):
            theta = random.uniform(np.pi/18,np.pi/2)
            roller = random.randrange(3)
            coords = np.dot(coords, self.rotation_matrix(theta,roller))
            for i in range(len(coords)):
                coord=coords[i]
                tmpx=abs(coord[0]-x)
                tmpy=abs(coord[1]-y)
                tmpz=abs(coord[2]-z)
                if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                    grid[j+1,np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]
                
        return grid

Feature = Feature_extractor()


#Get the path of training and test set
train_dirs = glob(os.path.join('trainset','*')) 
core_dirs = glob(os.path.join('coreset','*'))
core_dirs.sort()
core_id = [os.path.split(i)[1] for i in core_dirs]
train_new_dirs=[]
for i in train_dirs:
    pdb_id = os.path.split(i)[1]
    if pdb_id not in core_id:
        train_new_dirs.append(i)
np.random.shuffle(train_new_dirs)


#Get the data of affinties of proteins and ligands. -logKd/Ki values are the label.
affinity ={}
with open('INDEX_general_PL_data.2019','r') as f:
    for line in f.readlines():
        if line[0] != '#':
            affinity[line.split()[0]] = line.split()[3]
train_label=[]
core_label=[]
for i in train_new_dirs:
    pdb_id=os.path.split(i)[1]
    train_label.extend([affinity[pdb_id]]*10)
for i in core_dirs:
    core_id=os.path.split(i)[1]
    if not affinity.get(core_id):
        print(core_id)
    else:
        core_label.append(affinity[core_id])
train_label=np.array(train_label,dtype=np.float32)
core_label=np.array(core_label,dtype=np.float32)

#Save the label data of training and test set.
with open('/mnt/nfs/wangy/train_label.pkl','wb') as f:
    pickle.dump(train_label,f)

with open('/mnt/nfs/wangy/core_label.pkl','wb') as f:
    pickle.dump(core_label,f)


#Feature engineering of test set. 
core_complexes = []
for directory in core_dirs:
    pdb_id = os.path.split(directory)[1]
    ligand = next(pybel.readfile('mol2',os.path.join(directory,pdb_id+'_ligand.mol2')))
    pdb = next(pybel.readfile('pdb',os.path.join(directory,pdb_id+'_protein.pdb')))
    core_complexes.append((pdb,ligand))   

core_grids=None
for mols in core_complexes:
    coords1, features1 = Feature.get_features(mols[0],1)
    coords2, features2 = Feature.get_features(mols[1],0)
    
    center=(np.max(coords2,axis=0)+np.min(coords2,axis=0))/2
    coords=np.concatenate([coords1,coords2],axis = 0)
    features=np.concatenate([features1,features2],axis = 0)
    assert len(coords) == len(features)
    coords = coords-center
    grid=Feature.grid(coords,features,rotations=0)
    if core_grids is None:
        core_grids = grid
    else:
        core_grids = np.concatenate([core_grids,grid],axis = 0)
    
    with open('/mnt/nfs/wangy/core_grids.pkl','wb') as f:
        pickle.dump(core_grids, f)

#Feature engineering of training set.
train_complexes = []
for directory in train_new_dirs:
    pdb_id = os.path.split(directory)[1]
    ligand = next(pybel.readfile('mol2',os.path.join(directory,pdb_id+'_ligand.mol2')))
    pdb = next(pybel.readfile('pdb',os.path.join(directory,pdb_id+'_protein.pdb')))
    train_complexes.append((pdb,ligand))   

train_grids=None
for mols in train_complexes:
    coords1, features1 = Feature.get_features(mols[0],1)
    coords2, features2 = Feature.get_features(mols[1],0)
    
    center=(np.max(coords2,axis=0)+np.min(coords2,axis=0))/2
    coords=np.concatenate([coords1,coords2],axis = 0)
    features=np.concatenate([features1,features2],axis = 0)
    assert len(coords) == len(features)
    coords = coords-center
    grid=Feature.grid(coords,features)
    if train_grids is None:
        train_grids = grid
    else:
        train_grids = np.concatenate([train_grids,grid],axis = 0)
    
with open('/mnt/nfs/wangy/train_grids.pkl','wb') as f:
    pickle.dump(train_grids, f)
