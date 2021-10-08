#!/usr/bin/env python
# coding: utf-8

# In[6]:
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, Dense, Flatten, Activation, BatchNormalization
from openbabel import pybel
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[11]:
#Converts the protein-ligand complexes into 4D tensor. 
class Feature_extractor():
    def __init__(self):
        self.atom_codes = {}
        others = ([3,4,5,11,12,13]+list(range(19,32))+list(range(37,51))+list(range(55,84)))
        atom_types = [1,(6,1),(6,2),(6,3),(7,1),(7,2),(7,3),8,15,(16,2),(16,3),34,[9,17,35,53],others]
      
        for i, j in enumerate(atom_types):
            if type(j) is list:
                for k in j:
                    self.atom_codes[k] = i
                
            else:
                self.atom_codes[j] = i              
        
        self.sum_atom_types = len(atom_types)
        
    #Onehot encoding of each atomic type
    def encode(self, atomic_num, molprotein):
        encoding = np.zeros(self.sum_atom_types*2)
        if molprotein == 1:
            encoding[self.atom_codes[atomic_num]] = 1.0
        else:
            encoding[self.sum_atom_types+self.atom_codes[atomic_num]] = 1.0
        
        return encoding
    
    #Get coords and features 
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

    #Generate 4D tensor
    def grid(self,coords, features):
        assert coords.shape[1] == 3
        assert coords.shape[0] == features.shape[0]  
        
        grid=np.zeros((1,20,20,20,features.shape[1]),dtype=np.float32)
        x=y=z=np.array(range(-10,10),dtype=np.float32)+0.5
        for i in range(len(coords)):
            coord=coords[i]
            tmpx=abs(coord[0]-x)
            tmpy=abs(coord[1]-y)
            tmpz=abs(coord[2]-z)
            if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                grid[0,np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]                
        return grid


# In[13]:
def get_grid(protein, ligand):
    Feature = Feature_extractor()
    coords1, features1 = Feature.get_features(protein,1)
    coords2, features2 = Feature.get_features(ligand,0)
    
    center=(np.max(coords2,axis=0)+np.min(coords2,axis=0))/2
    coords=np.concatenate([coords1,coords2],axis = 0)
    features=np.concatenate([features1,features2],axis = 0)
    assert len(coords) == len(features)
    coords = coords-center
    grid=Feature.grid(coords,features)
    
    return grid


# In[26]:
def build_model():
    model = tf.keras.Sequential([
    Conv3D(7,kernel_size=(1,1,1),input_shape=(20,20,20,28),strides=(1,1,1)),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    Conv3D(7,kernel_size=(3,3,3)),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    Conv3D(7,kernel_size=(3,3,3)),
    BatchNormalization(),
    Activation(tf.nn.relu),
    Conv3D(28,kernel_size=(1,1,1)),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    Conv3D(56,kernel_size=(3,3,3),padding='same'),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    MaxPooling3D(pool_size=2),
    Conv3D(112,kernel_size=(3,3,3),padding='same'),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    MaxPooling3D(pool_size=2),
    Conv3D(224,kernel_size=(3,3,3),padding='same'),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    MaxPooling3D(pool_size=2),
    Flatten(),
    Dense(256),
    BatchNormalization(),
    Activation(tf.nn.relu),
    Dense(1)])

    model.load_weights('weights_22_112-0.0083.h5')
    return model

  
def predict(protein, ligand, model):
    grid = get_grid(protein, ligand)
    result = model.predict(grid) * 15
    return result


# In[1]:
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Predict the affinity of protein and ligand!')
    parser.add_argument('--protein', '-p', required=True, help='protein file')
    parser.add_argument('--pff', '-m', default='pdb', help='file format of protein')
    parser.add_argument('--ligand', '-l', required=True, help='ligand file')
    parser.add_argument('--lff', '-n', default='mol2', help='file format of ligand')
    parser.add_argument('--output', '-o', default='predicted_pKa.out', help='output file')
    args = parser.parse_args()

    ligands = list(pybel.readfile(args.lff,args.ligand))
    protein = next(pybel.readfile(args.pff,args.protein))
    model = build_model()
    with open(args.output,'w') as f:
        f.write('Predict the affinity of %s and %s\n' % (args.protein, args.ligand))
        for ligand in ligands:
            result = predict(protein, ligand, model)
            f.write('%.4f\n' % result)
    print('Done!')


if __name__ == '__main__':
    main()
