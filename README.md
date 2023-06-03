GEM-Screen

This repository contains the python codes for GEM-Screen, an active learning deep network model with geometry-enhanced molecular representation
to accelerate the ultralarge library docking. GEM-Screen utilizes GeoGNN architechture, which is in the PaddleHelix tool, to generate the geometry-enhanced
molecular descriptors. Thus one must first install the PaddleHelix tool as according to https://github.com/PaddlePaddle/PaddleHelix.

The following steps show how to build GEM-Screen model and predict docking scores.

1. Reference dataset preparation

cd GEM-Screen/apps/pretrained_compound/ChemRL/GEM/chemrl_downstream_datasets/qm7/raw

Prepare the compound dataset, which contains the compounds SMILES and corresponding docking scores, e.g. DOCK3.7, GLIDE SP scores, and write these
information into qm7.csv file.

2. Training model

cd GEM-Screen/apps/pretrained_compound/ChemRL/GEM
sh scripts/finetune_regr.sh

scripts/finetune_regr.sh is used to set training parameters like initial model, dataset, and epoch number.  One can run it to generate molecular descriptors in
catched_data, and train the model on the basis of an initial model. The results will be generated and showed in ./log/pretrain-qm7/model1/final_result, and
the parameters of the model generated at each epoch can be find in GEM-Screen/output/chemrl_gem/finetune/qm7/model1.  

3. Predicting

cd GEM-Screen/apps/pretrained_compound/ChemRL/GEM
sh scripts/pred_regr.sh

In predict.py, you can set the path of the finetuned model obtained from the training step in "build model" section. And in scripts/pred_regr.sh, you can set
the path to read dataset, and generate molecular descriptors in catched_data, which can be used to predict the docking scores.

The GEM-Screen models for AMPC and D4 systems are provided and ready to be used directly in GEM-Screen/apps/pretrained_compound/ChemRL/GEM/pretrain_models-chemrl_gem.


