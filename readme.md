command for processing crossdocked dataset:
```console
python data_processing/crossdocked/process_crossdocked.py --config=configs/dev_config.yml --index_file=/home/ian/projects/mol_diffusion/DiffSBDD/data/split_by_name.pt --data_dir=/home/ian/projects/mol_diffusion/DiffSBDD/data/crossdocked_pocket10 --output_dir=data/crossdock_processed
```

a comment about the dataset:
it looks like the dataset is crossdocked where the receptors already have waters removed and they've been reduced to an atom pocket of size around 10 angstrom. I followed the string of papers that use this dataset and they give a very paltry description of how exactly this dataset was produced from CrossDocked2020. This is infuriating. It also means that most of my "processing" code is useless because a pocket has already been selected and certain types of atoms already removed. Did they remove all metals from proteins? I don't know! Nobody has said anywhere. Should I include heavy metals in the approved receptor atom types? I don't know! I would have to manually check if any of those atoms appear in the dataset, which I don't have time to do. This is ridiculous. 

todo:
1. script arguments for recording a training run
    - specify output dir
    - give the training run a name
    - specify write/test intervals
    - also, we need to implement the test loop 
3. record train and test loss, save/load model weights
5. code for sampling a molecule, computing metrics?
6. process dataset for diffsbdd
7. train diffsbdd


ok so i set the remove_hydrogens flag to True, but apparently sometimes there are still hydrogens. but it seems quite rare so i'm actually going to ignore this for now..
there are a few "bugs" that occur - proteins that can't be parsed, proteins that can be parsed by for some reason they
end up having 0 pocket atoms