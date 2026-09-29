# Deep Research

A small PyTorch playground for experimenting with models and training loops.

## Setup

Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The package contains a deliberately small multilayer perceptron and a
single-batch training helper so experiments can start without boilerplate:

```python
import torch
from deep_research import PlaygroundMLP, train_step

model = PlaygroundMLP(input_size=2, hidden_size=8, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss = train_step(model, optimizer, torch.randn(16, 2), torch.randn(16, 1))
print(loss)
```

Run the tests with:

```bash
pytest
```