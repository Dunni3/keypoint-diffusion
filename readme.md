command for processing crossdocked dataset:
```console
python data_processing/crossdocked/process_crossdocked.py --config=configs/dev_config.yml --index_file=/home/ian/projects/mol_diffusion/DiffSBDD/data/split_by_name.pt --data_dir=/home/ian/projects/mol_diffusion/DiffSBDD/data/crossdocked_pocket10 --output_dir=data/crossdock_processed
```

todo:
1. create optimizer
    - gradient clipping?
    - exponential weight decay?
2. loss.backwards, optimizer.step
3. record train and test loss, save/load model weights
4. script arguments for num_epochs, batch_size
5. code for sampling using trained model
6. 