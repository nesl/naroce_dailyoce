import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from models import MambaModel
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union
from torch import Tensor


class NARMamba(nn.Module):
    def __init__(
    self, 
    mamba_config, 
    nar_vocab_size,
    out_cls_dim,
    ) -> None:
        super().__init__()
        d_model = mamba_config.d_model
        self.embedding = nn.Embedding(nar_vocab_size, d_model)

        nar_backbone = MambaModel(mamba_config)
        classifier_head = nn.Linear(d_model, out_cls_dim)
        self.nar = nn.Sequential(nar_backbone, classifier_head)

    def forward(self, x):
        x = self.embedding(x)
        x = self.nar(x)
        return x


class AdapterMamba(nn.Module):
    def __init__(
    self,
    mamba_config,
    dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = MambaModel(mamba_config)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x


class AdapterMLP(nn.Module):
    def __init__(
    self,
    input_dim,
    hidden_dim,
    n_layer,
    output_dim=128,
    dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_layer - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer (no dropout after final layer)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    



class NarocePipeline(nn.Module):
    def __init__(
    self, 
    frozen_nar,
    adapter_model,
    ) -> None:
        super().__init__()
        self.adapter = adapter_model
        self.nar = frozen_nar

    def forward(self, x):
        x = self.adapter(x)
        # Map into the latent space of NAR
        x = self.nar(x)
        return x


class StateNaroce(nn.Module):
    def __init__(self, 
                 mamba_config, 
                 nar_vocab_size: int, 
                 state_vocab_size: int, 
                 out_cls_dim: int):
        '''
        Mamba-based NAR with two output heads - one for CE label classification and the other for state classification
        '''
        super().__init__()
        d_model = mamba_config.d_model
        self.in_encoder = nn.Embedding(nar_vocab_size, d_model)
        self.state_encoder = nn.Embedding(state_vocab_size, d_model)
        
        self.nar_backbone = MambaModel(mamba_config)
        self.out_classifier = nn.Linear(d_model, out_cls_dim) # CE classifier
        self.state_classifier = nn.Linear(d_model, state_vocab_size)

    def forward(self, x, s, inference_params=None):
        x_embed = self.in_encoder(x)
        s_embed = self.state_encoder(s)

        h = self.nar_backbone(x_embed + s_embed, inference_params=inference_params)
        x = self.out_classifier(h)
        s = self.state_classifier(h)

        return x, s
    

@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()
 

if __name__=="__main__":
    # batch, length, dim = 2, 60, 128
    # x = torch.randn(batch, length, dim).to('cuda')

    # mamba_config = MambaConfig(d_model=128, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
    # model = MambaModel(mamba_config).to('cuda')
    # a = x[:, 0:1, :]
    # print(a.shape)
    # y = model(a)
    # print(x.shape, y.shape)
    # # assert y.shape == x.shape
    # @torch.inference_mode()
    # def run():
    #     batch, length, dim = 2, 60, 128
    #     x = torch.randn(batch, length, dim).to("cuda")

    #     # Training-style forward pass (full sequence in parallel)
    #     y1 = model(x)
    #     assert y1.shape == x.shape

    #     y4 = model(x)

    #     # Inference-style forward pass (full sequence in parallel)
    #     infer_params = InferenceParams(max_batch_size=batch, max_seqlen=length)
    #     y2 = model(x, inference_params=infer_params)

    #     # Inference-style forward pass (step by step using for loop)
    #     infer_params = InferenceParams(max_batch_size=batch, max_seqlen=length)
    #     outs = []
    #     for i in range(length):
    #         out = model(x[:, i : i + 1, :], inference_params=infer_params)
    #         infer_params.seqlen_offset += 1
    #         outs.append(out)
    #     y3 = torch.cat(outs, 1)

    #     print(torch.allclose(y1, y2, rtol=1e-4))  # prints True
    #     print(torch.allclose(y2, y3, rtol=1e-4))  # prints False
    #     print(torch.allclose(y1, y3, rtol=1e-4))  # prints False
    #     print(torch.allclose(y3[:, :10, :], y2[:, :10, :]))
    #     # print(torch.allclose(y1, y4)) 
    from mamba_ssm import Mamba2
    def run():
        batch, length, dim = 2, 64, 16
        x = torch.randn(batch, length, dim).to("cuda")
        model = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=64,  # SSM state expansion factor, typically 64 or 128
            d_conv=4,    # Local convolution width
            expand=4,    # Block expansion factor
            headdim=8,
            layer_idx=0
        ).to("cuda")

        # Training-style forward pass (full sequence in parallel)
        y1 = model(x)
        assert y1.shape == x.shape

        # Inference-style forward pass (full sequence in parallel)
        infer_params = InferenceParams(max_batch_size=batch, max_seqlen=length)
        y2 = model(x, inference_params=infer_params)

        # Inference-style forward pass (step by step using for loop)
        infer_params = InferenceParams(max_batch_size=batch, max_seqlen=length)
        outs = []
        for i in range(length):
            out = model(x[:, i : i + 1, :], inference_params=infer_params)
            infer_params.seqlen_offset += 1
            outs.append(out)
        y3 = torch.cat(outs, 1)

        print(torch.allclose(y1, y2))  # prints True
        print(torch.allclose(y2, y3, rtol=1e-2))  # prints False
        print(torch.allclose(y1, y3, rtol=1e-2))  # prints False
            
    run()


 