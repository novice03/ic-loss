## Inverse Contrastive Loss

Implementations of inverse contrastive loss from [Learning Invariant Representations using Inverse Contrastive Loss](https://arxiv.org/abs/2102.08343). The model architecture used on the ADNI dataset in the paper is also included along with PyTorch and Tensorflow implementations of the loss function.

## Installation

```bash
$ pip install ic-loss
```

## Usage

```python
import torch
from ic_loss.losses import icl, icl_tf # icl - pytorch, icl_tf - tensorflow
from ic_loss.models import ADNIResNet # ADNIResNet - pytroch model used in the paper

model = ADNIResNet()

x = torch.randn([1, 1, 512, 512])
logits, latent = model(x)

loss = icl(latent, c) # c - extraneous attribute 

```