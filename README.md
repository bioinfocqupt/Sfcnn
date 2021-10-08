# Sfcnn

A scoring function model based on 3D convolutional neural network for protein-ligand binding affinity prediction.

The model was trained on the PDBbind database version 2019 and tested on CASF-2016, CASF-2013, CSAR and axtex datasets.

The model can be applied for rescoring the docking results.

# Contact

Yu Wang, Chongqing Key Laboratory on Big Data for Bio Intelligence, Chongqing University of Posts and Telecommunications, Chongqing 400065, China; Email: wangyu@cqupt.edu.cn

# Dependencies and Installation

In order to use the model necessary packages should be installed

- anaconda3 (the best choice)

- tensorflow2

```shell
conda install tensorflow
```

- openbabel

```shel
conda install -c openbabel openbabel
```

In order to run the scripts you need more:

- jupyter-nootbook
- pandas
- matplotlib
- scikit-learn
- mayavi

Also, the corresponding dataset is required.

# Usage

###### ** Make sure the h5 weights file in your working path and 'predict.py' file has been downloaded correctly.**

###### Protein and ligand structure files must be provided separately and you can use any appropriate file format supported by openbabel.

This model is very easy to use and the protein-ligand binding affinity could be predicted directly and quickly. According to our summary, it can be used in three ways. There are details of their usage in the 'test' folder.

##### 1. If you have a single protein structure and a single ligand structure, use:

```shell
python predict.py -p 1a30_protein.pdb -l 1a30_ligand.mol2
python predict.py -p 1a30_protein.pdb -m pdb -l 1a30_ligand.mol2 -n mol2 -o output1
```

##### 2. If you have a single protein structure and many ligand structures in a single file just like the results of many docking software: 

```shell
python predict.py -p 1a30_protein.pdb -m pdb -l 1a30_decoys.mol2 -n mol2 -o output2
```

##### 3. If you have more complex tasks such as analysis of the results of large-scale virtual screening and predicting the affinity of various proteins and ligands on big data set, you best use the model as follows:

```python
import predict #step 1
from openbabel import pybel

model = predict.build_model() #step 2
protein = next(pybel.readfile('pdb','1a30_protein.pdb'))
ligand = next(pybel.readfile('mol2','1a30_ligand.mol2'))
result = predict.predict(protein, ligand, model) #step 3
```

Combining the model (predict.py ) and python scripts can handle a wide variety of prediction tasks.  And it only takes three steps to complete this.

Here is an example of predicting protein-ligand affinities on CASF-2016 dataset:

```python
import predict   #step 1
from openbabel import pybel

model = predict.build_model()   # step 2
core_dirs = glob(os.path.join('CASF-2016','CASF-2016','coreset','*'))
core_id = [os.path.split(i)[-1] for i in core_dirs]
f = open('output.csv','w')
f.write('#code\tscore\n')
for pdbid in core_id:
    proteinfile = os.path.join('CASF-2016','CASF-2016','coreset',pdbid, pdbid+ '_protein.pdb')
    ligandfile = os.path.join('CASF-2016','CASF-2016','coreset',pdbid, pdbid+'_ligand.mol2')
    protein = next(pybel.readfile('pdb',proteinfile))
    ligand = next(pybel.readfile('mol2',ligandfile))
    result = predict.predict(protein, ligand, model)  # step 3, can be combined with loop statements to predict multiple times
    f.write(pdbid+'\t%.4f\n' % result)
f.close()
```

