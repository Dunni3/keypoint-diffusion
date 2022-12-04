command for processing crossdocked dataset:
```console
python data_processing/crossdocked/process_crossdocked.py --config=configs/dev_config.yml --index_file=/home/ian/projects/mol_diffusion/DiffSBDD/data/split_by_name.pt --data_dir=/home/ian/projects/mol_diffusion/DiffSBDD/data/crossdocked_pocket10 --output_dir=data/crossdock_processed
```

todo:
1. script arguments for recording a training run
    - specify output dir
    - give the training run a name
    - specify write/test intervals
    - also, we need to implement the test loop 
2. remove hydrogens from dataset
3. record train and test loss, save/load model weights
5. code for sampling a molecule, computing metrics?
6. process dataset for diffsbdd
7. train diffsbdd