import torch

def test_rim():
    from RIM_pytorch import RIM
    from RIM_pytorch.depth_less_transformer import DepthlessTransformer

    model = DepthlessTransformer(512, num_blocks = 6)

    x = torch.randn(1, 1024, 512)
    out = model(x)

    assert out.shape == (6, *x.shape)
