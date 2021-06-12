# HyperTeNet
Hypergraph and Transformer-based Neural Network for Personalized List Continuation

#### Extract Dataset
To train or evaluate **HyperTeNet** on a dataset, extract the zipped
dataset in the [data](data/) directory.

------------
#### Training HyperTeNet
Use the following scripts to train/evaluate the model. For evaluation,
make sure that a trained model corresponding to a dataset is present in
the [saved_models](saved_models/) directory.


**1. Art of the Mix (AOTM)**  
*Train:* `python train.py --path="./data/aotm/" --dataset="aotm"
--num_epochs=300 --epoch_mod=5 --num_negatives=5`

*Evaluate:* `python train.py --path="./data/aotm/" --dataset="aotm"`

------------





!
