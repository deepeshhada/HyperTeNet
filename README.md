# HyperTeNet
Hypergraph and Transformer-based Neural Network for Personalized List Continuation

------------
### Extract Dataset
To train or evaluate **HyperTeNet** on a dataset, extract the zipped
dataset in the [data](data/) directory. The datasets can be downloaded
from [here](https://drive.google.com/drive/folders/1ravjFWBgUb_cgpn2Z00ELKyY3CNQXqiv?usp=sharing).

------------
### Training and Evaluation
Use the following scripts to train/evaluate the model. For evaluation,
make sure that a trained model corresponding to a dataset is present in
the [saved_models](saved_models/) directory.


**1. Art of the Mix (AOTM)**

*Train:* `python train.py --path="./data/aotm/" --dataset="aotm"
--num_layers=2 --num_epochs=300 --num_negatives=5 --num_negatives_seq=2
--lr=0.001`

*Evaluate:* `python eval.py --path="./data/aotm/" --dataset="aotm" --num_layers=2`

**2. Goodreads**

*Train:* `python train.py --path="./data/goodreads/" --dataset=goodreads"
--num_layers=3 --num_epochs=300 --num_negatives=3 --num_negatives_seq=5
--lr=0.0008`

*Evaluate:* `python eval.py --path="./data/goodreads/" --dataset="goodreads"
--num_layers=3`

**3. Spotify**

*Train:* `python train.py --path="./data/spotify/" --dataset=spotify"
--num_layers=3 --num_epochs=300 --num_negatives=3 --num_negatives_seq=5
--lr=0.0008`

*Evaluate:* `python eval.py --path="./data/spotify/" --dataset="spotify"
--num_layers=3`

**4. Zhihu**

*Train:* `python train.py --path="./data/zhihu/" --dataset="zhihu"
--num_layers=3 --num_epochs=300 --num_negatives=5 --num_negatives_seq=6
--lr=0.002`

*Evaluate:* `python eval.py --path="./data/zhihu/" --dataset="zhihu" --num_layers=3`

------------
