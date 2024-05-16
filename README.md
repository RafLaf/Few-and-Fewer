# Few and Fewer: Learning Better from Few Examples Using Fewer Base Classes
This repository is the official implementation of **Few and Fewer: Learning Better from Few Examples Using Fewer Base Classes** currently under review.

## Supplementary material
![alt text](https://github.com/RafLaf/Few-and-Fewer/blob/main/supp_mat/all_supp_mat.png?raw=true)



## How to install 
This code uses the task2vac, to install it run the following commands:

    cd <path> # First decide where to clone the repository
    git clone https://github.com/awslabs/aws-cv-task2vec.git # Clone the repository
    export PYTHONPATH=<path>/aws-cv-task2vec:$PYTHONPATH # Then add it to the path

The install the requirements

    pip install -r requirements.txt
## Download the data
Follow [DATASETS.md](datasets/DATASETS.md) to install the datasets.


## Run the experiments

First set the path to save data

    export DATASET_DIR=<path>

Go to bash_script file

    cd bash_scripts

Choose which dataset to run the experiments on
    
    DATASET_OPTION=<dataset>

<b>Available dataset</b>



| |  |  | | |  |  | |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|aircraft | cub| dtd|fungi|omniglot|mscoco|traffic_signs|vgg_flower|

Choose a mode to run with 

    export MODE=<mode>

<b>Available modes</b>

| Name | MODE |
|:--------:|:--------:|
|1-shot 5-ways| 1s5w |
|5-shot 5-ways| 5s5w | 
|Metadataset protocol | MD |

Finally, to run the experiment, run one of the following scripts based on the desired method:

<b>Available options</b>

| Name   | Method | Script |
|:--------:|:--------:|:--------:|
|Fine tuning on support set| S | ./B/measure.sh|
|Domain Informed | DI | ./DI/finetune.sh; ./DI/measure.sh|
|Task Informed | TI | ./TI/classifier.sh |
|Task agnostic | TA | ./TA/classifier.sh; ./TA/finetune.sh; TA/measure.sh|
|IDB | IDB | ./TA/classifier.sh; ./TA/finetune.sh; TA/measure.sh|



