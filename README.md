# 3D Traffic Flow

## Usage Example

### Train

```bash
python -m train.train_mdm --save_dir ~/tmp_line/ --overwrite --dataset line --param_constrained --arch informer
```

### Generate

```bash
python -m sample.generate_line --model_path ~/tmp_line/model000030015.pt --output_dir ~/tmp_line_result/ --num_repetitions 1 --params_for_line 100,50000,50000,20000,0,0,0
```

