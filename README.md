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

make llama-dense postfix=$tag
```

### Tests (comming soon)
```bash
pytest -v --disable-warnings moe-lab/tests/
