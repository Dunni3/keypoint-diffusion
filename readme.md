# **Accelerating Inference in Molecular Diffusion Models with Latent Representations of Protein Structure.**

![Keypoint-Conditioned Diffusion](https://github.com/Dunni3/keypoint-diffusion/assets/29707787/9dd7cbc4-1d10-4843-9cd5-d6565d088811)

# Setting up dev environment

All code in this repository was run on python 3.10. In my experience, installing with python 3.11 caused dependency conflicts. I recommend using conda or mamba to create a python 3.10 environment. You should install the following dependencies in this order. You can copy and paste this code block into a shell script for easy use.

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

For the paper all models are trained on the bindingmoad dataset. We use the same splits of bindingmoad as in [DiffSBDD](https://github.com/arneschneuing/DiffSBDD). Clone the DiffSBDD repo, and follow the directions for downloadng the bindingmoad dataset into the `data/` folder of the DiffSBDD Repo. You don't need to run their scripts for processing the data, you just need to download it.

You will notice that there is some code in this repository for processing/working with the crossdocked dataset. The results in the paper only focus on models trained on the bindingmoad dataset. Future releases of this repository / versions of this paper may include results on the crossdocked dataset. However, for now, the crossdocked dataset is not used for training or evaluation, and the processing scripts are not guaranteed to work.

# How we define a model (config files)

Specifications of the model and the data that the model is trained on are all packaged into one config file. The config files are just yaml files. Once you setup your config file, you pass it as input to the data processing scripts in addition to the training scripts. An example config file is provided at `configs/dev_config.yml`. This example config file also has some helpful comments in it describing what the different parameters mean.

Actual config files used to train models presented in the paper are available in the `trained_models/` directory.

Note, you don't have to reprocess the dataset for every model you train, as long as the models you are training contain the same parameters under the `dataset` section of the config file. 


# A note on understanding our scripts

All of the steps of training, sampling, and evaluation are run through various scripts in this repo. In this readme, I describe in words the inputs that are provided to each script. Each of these scripts implements command line arguments via the argparse library. You can always run `python <script_name>.py --help` to see a list of command line arguments that the script accepts. You can also just open the script and inspect the `parse_args()` function to see what command line arguments are accepted.

# Processing the data

There are two scripts in this repository for processing the crossdocked and bindingmoad datasets so they can be ingested by our dataset class at training/evaluation time. These are `process_crossdocked.py` and `process_bindingmoad.py`. Unfortunately, `process_crossdocked.py` might be broken at the moment. 

<!-- command for processing crossdocked dataset:
```console
python process_crossdocked.py --config=configs/dev_config.yml --index_file=/home/ian/projects/mol_diffusion/DiffSBDD/data/split_by_name.pt --data_dir=/home/ian/projects/mol_diffusion/DiffSBDD/data/crossdocked_pocket10 --output_dir=data/crossdock_processed
``` -->

command for processing bindingmoad dataset
```console
python process_bindingmoad.py --config_file=configs/dev_config.yml --data_dir=/home/ian/projects/mol_diffusion/DiffSBDD/data/ 
```

## Output of data processing

Running either `process_crossdocked.py` or `process_bindingmoad.py` will write a processed version of the dataset to whatever directory was specified by the `--output_dir` command-line argument. 

# Training

Models are trained using the `train.py` script. You pretty much only have to provide a config file to `train.py` using the `--config` option. You can also override certain parameters in the config file via command-line arguments to the training script. However, this functionality was only implemented for weights and biases hyperparameters sweeps and as such, not all model hyperparameters defined in the config files are exposed via command-line arguments. 

## Training output

Within a config file, the parameter `results_dir` in the `experiment` section specifies where the results of training will be written.
Each time `train.py` is run, a new directory will be created within `results_dir` with a timestamped name. Within this directory will be the model's config file, pickled versions of the model, and a pickle file containing logs of the training and validation metrics.

# Sampling molecules for any given protein (BYOP)

The script `byop.py` (bring your own protein) accepts as input: the location of a trained model, a pdb defining a protein structure, and an sdf file specifying a "reference ligand" which is located inside the protein binding pocket. This reference ligand need not actually be a ligand. The script will simply use the coordinates of the reference ligand to define the binding pocket and it will start sampling molecules by placing atoms at the center of mass of the reference ligand. 

`byop.py` will do a force-field minimization of generated ligands inside the binding pocket if you specify the `--pocket_minimization` flag.

# Sampling molecules and visualizing the diffusion process.

Use the script `sample.py` with the `--visualize` flag. Ideally, the visualize feature should be incorporated into `test.py` because these scripts are doing almost the exact same thing, iirc. Note that this only allows you to visualize the sampling process for proteins in the original dataset. I haven't yet implemented sampling visualization with `byop.py`.

# Sampling molecules for evaluation of model performance.

The script `test.py` will sample ligands for ligand-receptor complexes contained in the test/validation sets. You just need to point it to an output directory produced by `train.py` with the `--model_dir` flag. The test script will then load the last-saved model state. You can also instead specify a specific model checkpoint to use via the `--model_file` flag. 

Note that `test.py` assumed that the processed dataset specified by the models's config file exists. This also means that `test.py` can only sample complexes from the original dataset. If you want to sample ligands for any protein, see the section on BYOP above.

`test.py` will write the sampled molecules to the directory specified by the `--output_dir` flag. This directory will contain a subdirectory `sampled_mols` which itself contains a directory for each pocket that is sampled. The structure of these subdirectories is very specific and this particular structure is required for the downstream evaluation scripts to work.

# Evaluation pipeline for results presented in the paper

## Force-Field Minimization inside Binding Pocket

There are two options for doing ff-minimization of generated molecules inside the binding pocket:
1. You can do it while sampling molecules by passing the `--pocket_minimization` flag to `test.py`. This will cause `test.py` to write the minimized molecules into the same directory as the raw sampled molecules.
2. You can do it after sampling molecules by running the script `analysis/pocket_minimization.py`. This script accepts as input a directory of sampled molecules produced by `test.py`.

To elbaorate on the second option here: the second method is advantageous if you are minimizing many molecules across many pockets because you can parallelize the minimization across a cluster. The force-field minimzation of a single binding pocket is performed by the script `analysis/pocket_minimization.py`. You can run the script `gen_pocket_min_cmds.py` and pass it a `sampled_mols` directory produced by `test.py`, and then it will generate a shell script where each line of the shell script calls `pocket_minimization.py` on a different pocket. You can then run this shell script to run the force-field minimization on all of the sampled pockets. You could also use this shell script to construct an array job to parallelize minimization across a cluster. For a given pocket directory produced by `test.py`, the minimization script `analysis/pocket_minimization.py` will write the minimized molecules into that same directory as well as a csv containing the RMSDs of molecules before and after force-field minimization. 

## Scoring with AutoDock Vina

We actually use [gnina](https://github.com/gnina/gnina) to do molecule scoring. gnina returns Autodock Vina scores on molecules in addition to scores from its own scoring function. So to use our scoring pipeline, you will need to install gnina. 

The script `gen_docking_cmds.py` accepts as input a directory of sampled molecules produced by `test.py` and produces as output a text file containing list of gnina commands which then can be run as a shell script or array jobs across a cluster. By default, the commands generated will completely redock the reference ligand. This is not what we do for the paper. For the paper, we use the `--minimize` flag to generate commands that will simply minimize the ligand pose with respect to the Vina scoring function. 

We consider complete re-docking to be an unreliable/inaccurate indicater of generative model performance, as it can hide/obscure failures of the model to respect the geometry of the binding pocket and protein-ligand interactions. This idea is supported by the results of the [PoseChecker paper](https://arxiv.org/abs/2308.07413).

## Additional Metrics

Additional metrics of ligand quality (validity, QED, SA, per-pocket diversity) can be computed using the `compute_metrics.py` script. This script accepts as input a directory of sampled molecules produced by `test.py` and will write a pickle file into the parent directory of the `sampled_mols` directory. This pickle file contains a dictionary with all of these metrics.


<!-- # Some commands for producing publication reusts

## sampling validation set
```console
python gen_test_commands.py test_cmds.txt --filenames_file data/bindingmoad_refactor/val_filenames.pkl --lines 28 29 30 31 32 33 35 36 38 40 41 42 43 44 45 46
sbatch --array 1-3904 test_parallel.slurm
```

## running force-field minimization on sampled molecules
```console
python make_minimize_cmds.py
./pub_samples_min_cmds/py_cmds.sh
./pub_samples_min_cmds/sbatch_cmds.sh
```

## running vina scoring on ff minimized molecules
```console
python gen_docking_cmds.py pub_samples_2/ --exclude diffsbdd_ca_cond diffsbdd_ca_inpaint diffsbdd_full_cond --minimize
sbatch --array 1-3954 dock_cpu.slurm
```

```console
python gen_docking_cmds.py pub_samples_2/ --model_name validation_set --minimize
sbatch --array 1-244 dock_cpu.slurm -->
<!-- ``` -->

# Cleanup todo
- [ ] add license
- [x] create directory with trained models
- [x] write script for sampling molecules from trained models
- [x ] change name of directory and name of LigandDiffuser class
- [ ] add necessary dataset files for alpha-carbon only models
- [ ] add picture of ligand generation to readme
