command for processing crossdocked dataset:
```console
python process_crossdocked.py --config=configs/dev_config.yml --index_file=/home/ian/projects/mol_diffusion/DiffSBDD/data/split_by_name.pt --data_dir=/home/ian/projects/mol_diffusion/DiffSBDD/data/crossdocked_pocket10 --output_dir=data/crossdock_processed
```

command for processing bindingmoad dataset
```console
python process_bindingmoad.py --config_file=configs/dev_config_moad.yml --data_dir=/home/ian/projects/mol_diffusion/DiffSBDD/data/ 
```

a comment about the dataset:
it looks like the dataset is crossdocked where the receptors already have waters removed and they've been reduced to an atom pocket of size around 10 angstrom. I followed the string of papers that use this dataset and they give a very paltry description of how exactly this dataset was produced from CrossDocked2020. This is infuriating. It also means that most of my "processing" code is useless because a pocket has already been selected and certain types of atoms already removed. Did they remove all metals from proteins? I don't know! Nobody has said anywhere. Should I include heavy metals in the approved receptor atom types? I don't know! I would have to manually check if any of those atoms appear in the dataset, which I don't have time to do. This is ridiculous. 

todo:
1. add files/directions so data downloading/processing is self-contained 

it seems i can use the `nn.Module.model_parameters()` method to find the params just for rec encoder module which will let me specify different learning schedules for these params separately i.e., including 
the balance of losses into the learning rate scheduling algo

# sampling molecule from the test set

```console
python sample_crossdocked.py --output_dir=sample_epoch_130 --model_dir=experiments/baseline_clipping_1218162647 --n_replicates=4 --n_complexes=4
```