import pytest
import torch

param = pytest.mark.parametrize

@param('causal', (False, True))
def test_rim(
    causal
):
    from RIM_pytorch import RIM
    from RIM_pytorch.depth_less_transformer import DepthlessTransformer

    model = DepthlessTransformer(512, causal = causal, num_blocks = 6, num_tokens = 256)

    x = torch.randn(1, 1024, 512)
    logits = model(x)

    assert logits.shape == (1, 1024, 256)
