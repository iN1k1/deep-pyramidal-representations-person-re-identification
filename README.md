# PyrNet
This repo contains the source codes implemented to run the experiments for **person re-identification** used within the paper: "*Aggregating Deep Pyramidal Representations for Person Re-Idenfitication*", published in International Conference on Computer Vision and Pattern Recognition - Workshop on Target Re-identification and Multi-Target Multi-Camera Tracking, 2019.

# Data
The repository does not contain the datasets. You can download a copy of the Market-1501 dataset from here: [Datasets](https://drive.google.com/file/d/1HfgS3HveeY74Jz5rnTIrKB1eH4AIGwHg/view?usp=sharing). Just extract the zip within the "data" folder. To make the scripts running with other datasets (e.g., Duke, CUHK, etc.), you can just copy the original files with the same "data" folder.

# Usage
The solution has been written using the PyTorch framework and tested with the version specified with the *requirements.txt* file. If you want, feel free to run 
`pip install -r requirements.txt`
to get all the dependencies in place.
  
After that, you can just run 

    python main.py
to perform a train/test with single shot on the Market-1501 dataset (provided you have downloaded and copied it as described before).
If you want to test the solution with a different configuration, please have a look at the arguments within the `main.py` file. Those should be self-explanatory.

# Thanks
If you use the code contained in this package we appreciate if you'll cite our work.
> BIBTEX: @inproceedings{Martinel2019a,
author = {Martinel, Niki and Foresti, Gian Luca and Micheloni, Christian},
booktitle = {International Conference on Computer Vision and Pattern Recognition Workshops},
title = {{Aggregating Deep Pyramidal Representations for Person Re-Identification}},
year = {2019}
}