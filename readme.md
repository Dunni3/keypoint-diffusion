# Setting up dev environment

Create a conda/mamba environment. You should install the following dependencies in this order. You can copy and paste this code block into a shell script for easy use.

```console
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install pytorch-cluster pytorch-scatter -c pyg -y
mamba install -c dglteam/label/cu118 dgl -y
mamba install -c conda-forge rdkit -y
mamba install -c conda-forge openbabel -y
mamba install -c conda-forge pandas -y
mamba install -c conda-forge biopython -y
```

I might be missing a few dependencies. Feel free to add them here.

# Getting the data

We use the same datasets as [DiffSBDD](https://github.com/arneschneuing/DiffSBDD). Clone the DiffSBDD repo, and follow the directions for downloadng the dataets into the `data/` folder of the DiffSBDD Repo. You don't need to run their scripts for processing the data, you just need to download it.

# How we define a model (config files)

Specifications of the model and the data that the model is trained on are all packaged into one config file. The config files are just yaml files. Once you setup your config file, you pass it as input to the data processing scripts in addition to the training scripts. An example config file (the one i use for development) is provided at `configs/dev_config.yml`.

Note, you don't have to reprocess the dataset for every model you train, as long as the models you are training contain the same parameters under the `dataset` section of the config file. 

# Processing the data

There are two scripts in this repository for processing the crossdocked and bindingmoad datasets so they can be ingested by our dataset class at training/evaluation time. These are `process_crossdocked.py` and `process_bindingmoad.py`. Unfortunately, `process_crossdocked.py` might be broken at the moment. When building this repo I first wrote `process_crossdocked.py`. But then later I wrote `process_bindingmoad.py`, and from that point on all development was done using only the bindingmoad dataset, which included refactors of the data processing methods. Example commands for processing the datasets using these scripts are provided below. 

command for processing crossdocked dataset:
```console
python process_crossdocked.py --config=configs/dev_config.yml --index_file=/home/ian/projects/mol_diffusion/DiffSBDD/data/split_by_name.pt --data_dir=/home/ian/projects/mol_diffusion/DiffSBDD/data/crossdocked_pocket10 --output_dir=data/crossdock_processed
```

command for processing bindingmoad dataset
```console
python process_bindingmoad.py --config_file=configs/dev_config_moad.yml --data_dir=/home/ian/projects/mol_diffusion/DiffSBDD/data/ 
```

## Output of data processing

Running either `process_crossdocked.py` or `process_bindingmoad.py` will write a processed version of the dataset to whatever directory was specified by the `--output_dir` command-line argument. 

<!-- a comment about the dataset:
it looks like the dataset is crossdocked where the receptors already have waters removed and they've been reduced to an atom pocket of size around 10 angstrom. I followed the string of papers that use this dataset and they give a very paltry description of how exactly this dataset was produced from CrossDocked2020. This is infuriating. It also means that most of my "processing" code is useless because a pocket has already been selected and certain types of atoms already removed. Did they remove all metals from proteins? I don't know! Nobody has said anywhere. Should I include heavy metals in the approved receptor atom types? I don't know! I would have to manually check if any of those atoms appear in the dataset, which I don't have time to do. This is ridiculous.  -->

# Training

Models are trained using the `train_crossdocked.py` script. Which, despite the name, can train models on either the crossdocked or bindingmoad dataset. You pretty much only have to provide a config file to `train_crossdocked.py` using the `--config` option. You can also override certain parameters in the config file via command-line arguments to the training script. However, this functionality was only implemented for weights and biases hyperparameters sweeps and as such, not all model hyperparameters defined in the config files are exposed via command-line arguments. 

# Sampling molecules for evaluation of model performance.

Use the script `test_crossdocked.py` which, despite the name, can be used for both the bindingmoad and crossdocked datasets. 

# Sampling molecules and visualizing the diffusion process.

Use the script `sample_crossdocked.py` with the `--visualize` flag. Ideally, the visualize feature should be incorporated into `test_crossdocked.py` because these scripts are doing almost the exact same thing, iirc. 