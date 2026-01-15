import pytest
import torch
from moelab.utils.tensor_meter import TensorMeter


def test_init():
    meter = TensorMeter()
    assert meter.n == 0
    assert meter.last is None
    assert meter.avg is None


def test_single_update():
    meter = TensorMeter()
    x = torch.tensor([1.0, 2.0, 3.0])
    meter.update(x)
    
    assert meter.n == 1
    assert torch.allclose(meter.last, x)
    assert torch.allclose(meter.avg, x)


def test_multiple_updates():
    meter = TensorMeter()
    x1 = torch.tensor([1.0, 2.0, 3.0])
    x2 = torch.tensor([3.0, 4.0, 5.0])
    x3 = torch.tensor([5.0, 6.0, 7.0])
    
    meter.update(x1)
    meter.update(x2)
    meter.update(x3)
    
    assert meter.n == 3
    assert torch.allclose(meter.last, x3)
    assert torch.allclose(meter.avg, torch.tensor([3.0, 4.0, 5.0]))


def test_running_mean():
    meter = TensorMeter()
    values = [torch.tensor([float(i)]) for i in range(1, 11)]
    
    for val in values:
        meter.update(val)
    
    assert meter.n == 10
    assert torch.allclose(meter.avg, torch.tensor([5.5]))


def test_reset():
    meter = TensorMeter()
    meter.update(torch.tensor([1.0, 2.0, 3.0]))
    meter.reset()
    
    assert meter.n == 0
    assert meter.last is None
    assert meter.avg is None


def test_no_grad():
    meter = TensorMeter()
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    meter.update(x)
    
    assert not meter.last.requires_grad
    assert not meter.avg.requires_grad


def test_detach():
    meter = TensorMeter()
    x = torch.tensor([1.0], requires_grad=True)
    meter.update(x)
    
    # Modify original, meter should be unaffected
    x.data.fill_(99.0)
    assert torch.allclose(meter.last, torch.tensor([1.0]))
