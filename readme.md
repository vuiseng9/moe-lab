## moe-lab

### Install
```bash
# user mode
pip install -e .
# developer mode
pip install -e .[dev]
```

### Run
```bash
tag="preview"

make olmoe_no_lb postfix=$tag
make olmoe_lb_penalty postfix=$tag

make dsv3_no_lb postfix=$tag
make dsv3_lb_bias postfix=$tag

# just add sweep_lr=1 to enable learning rate sweep 
# analyze results
# modify lr in Makefile accordingly 
# (intentional design to be static instead of dynamic (too much runtime parameter, reduce mental load, and less log digging))
# lr is currently to my best found value.
```

### Tests (comming soon)
```bash
pytest -v --disable-warnings moe-lab/tests/
