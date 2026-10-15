import torch

from loopnav_finetune_v2.model import LoRALinear


def test_lora_is_initially_identity_and_has_gradients():
    torch.manual_seed(0)
    base = torch.nn.Linear(5, 7, bias=False)
    value = torch.randn(3, 5)
    expected = base(value).detach()
    wrapped = LoRALinear(base, rank=2, alpha=2.0, dropout=0.0)
    actual = wrapped(value)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert wrapped.base.weight.grad is None
    assert wrapped.lora_B.weight.grad is not None
