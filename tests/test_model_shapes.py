import torch


def test_resnet1d_forward_shape():
    from experiments.models.resnet1d import build_resnet1d

    model = build_resnet1d(in_channels=1, num_classes=1)
    x = torch.randn(2, 1, 2000)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 1)
