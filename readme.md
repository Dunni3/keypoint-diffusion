command for processing crossdocked dataset:
```console
python data_processing/crossdocked/process_crossdocked.py --config=configs/dev_config.yml --index_file=/home/ian/projects/mol_diffusion/DiffSBDD/data/split_by_name.pt --data_dir=/home/ian/projects/mol_diffusion/DiffSBDD/data/crossdocked_pocket10 --output_dir=data/crossdock_processed
```

a comment about the dataset:
it looks like the dataset is crossdocked where the receptors already have waters removed and they've been reduced to an atom pocket of size around 10 angstrom. I followed the string of papers that use this dataset and they give a very paltry description of how exactly this dataset was produced from CrossDocked2020. This is infuriating. It also means that most of my "processing" code is useless because a pocket has already been selected and certain types of atoms already removed. Did they remove all metals from proteins? I don't know! Nobody has said anywhere. Should I include heavy metals in the approved receptor atom types? I don't know! I would have to manually check if any of those atoms appear in the dataset, which I don't have time to do. This is ridiculous. 

todo:
1. test code for sampling a molecule. whats the point of p(x,h|z0)?
2. computing metrics?
3. issue with ligand COM. maybe we should always start at pocket COM?
4. expose all model hyperparameters to config file
5. learning rate schedules. training only REM before training diffusion model.
6. xavier uniform initialization in receptor encoder
7. visualize sampled molecules and keypoint location

data processing woes:
ok so i set the remove_hydrogens flag to True, but apparently sometimes there are still hydrogens. but it seems quite rare so i'm actually going to ignore this for now..
there are a few "bugs" that occur - proteins that can't be parsed, proteins that can be parsed by for some reason they
end up having 0 pocket atoms

it seems i can use the `nn.Module.model_parameters()` method to find the params just for rec encoder module which will let me specify different learning schedules for these params separately i.e., including 
the balance of losses into the learning rate scheduling algo

expose use_tanh hyperparameter to config file!!