# Parcellotron
![Parcellotron GUI](https://dl.dropboxusercontent.com/u/13850642/PARCELLOTRON.gif)

Introduction
------------
Connectivity-Based Parcellation (CBP) has become one of the most interesting application of MRI-based tractography [1], [2], [3], [4]. So far however performing CBP required extensive coding and technical skills.

The Parcellotron is a Connectivity-Based Parcellation software for the rest of us. It enables every researcher having access to tractography data to perform CBP from an intuitive graphical user interface, as well as from a command-line interface.

Installation
------------
Follow this [link](http://github.com/chrisfoulon/Parcellotron) to clone or download the repository.
```
$ git clone http://github.com/chrisfoulon/Parcellotron
```

Input data
--------------
Currently the Parcellotron accepts two kinds of input:

1. **Tractography 4D** - One 4D Nifti file containing the structural connectivity maps for each ROI in the seed region
2. **Tractography matrix** - The omatrix1 and omatrix3 output by [FSL Fdt](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide)  


Usage
-----
We explain here how to perform CBP using the omatrix1 output by FSL Fdt. A tutorial on the 4D format is coming soon.

1. Perform the probabilistic tractography from every voxel inside a mask containing both the seed region you intend to parcellate, and the target region (e.g the whole brain). We will soon post a sample script.
 
2. Create a folder structure similar to the one below. In particular make sure that: 
* the omat1 directory output by the tractography is contained in a folder named *Tracto_mat*
* the *Tracto_mat* subfolder also contains a Nifti image of the seed region - ending in *seedMask.nii.gz* and that of the target region ending with *targetMask* 

![Input_folder](https://dl.dropboxusercontent.com/u/13850642/folder_structure.png)

3. Launch the Parcellotron with:
```
$ python3.6 parcellotron.py
```

4. Follow the instructions in the GUI and choose the appropriate parameters according to your data type and hypotheses. 

Parameters (which are not self-explanatory)
----------------
* **Prefix of seed and target files**: (**Coming soon**. For now just use the names *seedMask* and *targetMask* with no prefixes)
* **Modality**: Refer to the *Input data* section above.
*  **Size of the ROIs**: in cubic mm. The number of seed voxels in each ROI will be rounded to fit the size specified here.



References
---------------
[1]: Johansen-Berg, H, Behrens, TE, Robson, MD, Drobnjak, I, Rushworth, MF, Brady, JM, Smith, SM, Higham, DJ, Matthews, PM (2004) Changes in connectivity profiles define functionally distinct regions in human medial frontal cortex. Proc Natl Acad Sci U S A, 101:13335–13340.

[2]: Thiebaut de Schotten, M, Urbanski, M, Batrancourt, B, Levy, R, Dubois, B, Cerliani, L, Volle, E (2016) Rostro-caudal Architecture of the Frontal Lobes in Humans. Cereb Cortex.

[3]: Cerliani, L, D’Arceuil, H, Thiebaut de Schotten, M (2016) Connectivity-based parcellation of the macaque frontal cortex, and its relation with the cytoarchitectonic distribution described in current atlases. Brain Struct Funct.

[4]: Cloutman, LL, Lambon Ralph, MA (2012) Connectivity-based structural and functional parcellation of the human cortex using diffusion imaging and tractography. Front Neuroanat, 6:34.


> Written with [StackEdit](https://stackedit.io/).
