import torch
import pytest

from deep_research import PlaygroundMLP, train_step


def test_model_returns_expected_shape():
    model = PlaygroundMLP(input_size=3, hidden_size=4, output_size=2)

    assert model(torch.ones(5, 3)).shape == (5, 2)


def test_invalid_layer_size_is_rejected():
    with pytest.raises(ValueError):
        PlaygroundMLP(input_size=0, hidden_size=4, output_size=2)


def test_train_step_updates_model():
    torch.manual_seed(0)
    model = PlaygroundMLP(1, 4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.tensor([[1.0], [2.0]])
    targets = 2 * inputs
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_step(model, optimizer, inputs, targets)

    assert loss >= 0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )
