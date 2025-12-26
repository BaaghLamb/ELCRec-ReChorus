# ELCRec: End-to-end Learnable Clustering for Intent Learning in Recommendation
SYSU ML course task: Replicating the ELCRec model on the ReChorus framework


This repository contains the implementation of **ELCRec: End-to-end Learnable Clustering for Intent Learning in Recommendation**, a model designed to enhance recommendation performance by integrating end-to-end learnable clustering for user intent learning. 
The model has been successfully reproduced within the **ReChorus** framework, which provides a unified environment for evaluating various recommendation algorithms.

You can find the ELCRec repository here: [ELCRec Repository](https://github.com/yueliu1999/ELCRec).
Check out the framework here: [ReChorus Repository](https://github.com/THUwangcy/ReChorus).

## Our Modifications

In our implementation of the **ELCRec** model, we made the following key modifications to adapt it to the ReChorus framework:

1. We **added** the `ELCRecRunner`module, which aligns with ReChorus's runner design pattern to handle training, evaluation, and inference workflows.

2. The core implementation of the **ELCRec** model is located in the `ReChrous/src/models/sequential/ELCRec.py` file, where the model architecture is re-implemented to fit ReChorus's data pipeline.

## Getting Started

To get started with **ELCRec** in the **Rechorus** framework, follow these steps:

1. Download or Clone this repository:
   ```bash
   git clone https://github.com/BaaghLamb/ELCRec-ReChorus.git
   ```
2. Install the required dependencies:
   ```bash
   cd ReChorus
   pip install -r requirements.txt
   ```
3. Prepare the datasets

4. Train and evaluate the model:
   ```bash
   cd ReChorus
   cd src
   python main.py --model_name ELCRec --dataset "Grocery_and_Gourmet_Food" --epoch 300 --batch_size 256 --emb_size 64 --num_layers 2 --num_heads 4 --num_intent_clusters 128 --temperature 0.1 --contrast_weight 0.1 --cluster_weight 0.1 --fusion_type add --lr 1e-3 --l2 1e-6 --early_stop 40 --history_max 20 --num_workers 0 --dropout 0.1 --save_final_results 1 --topk 5,10,20 --use_elcm True --use_icl True
   ```


