import torch
import torch.nn as nn
import tokenizers
import transformers
from transformers import AutoModel, AutoConfig
from utils import get_logger
from pathlib import Path
from overall_class import CFG
from typing import Dict

LOGGER = get_logger()  

def freeze(module):
    """
    Freezes module's parameters.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_freezed_parameters(module):
    """
    Returns names of freezed parameters of the given module.
    """
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
    return freezed_parameters

class AttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )  

    def forward(self, x, mask):
        w = self.attention(x).float()
        w[mask == 0] = float("-inf")
        w = torch.softmax(w, 1)
        x = torch.sum(w * x, dim=1)
        return x


class CustomModel(nn.Module):
    def __init__(
        self, 
        cfg, 
        config_path, 
        pretrained
    ):
        super().__init__()
        self.cfg = cfg
        self.window_size = cfg.window_size
        self.edge_len = cfg.edge_len
        self.except_edge_len = self.window_size - self.edge_len
        self.n_target = len(cfg.target_cols)

        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                cfg.model, output_hidden_states=True
            )
            self.config.hidden_dropout = 0.0
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_dropout = 0.0
            self.config.attention_probs_dropout_prob = 0.0
            LOGGER.info(self.config)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if cfg.multi_sample_dropouts is not None:
            self.dropouts = nn.ModuleList(
                [nn.Dropout(p) for p in cfg.multi_sample_dropouts]
            )
        else:
            self.dropouts = None

        pool_list = []
        for _ in range(self.n_target):
            pool = AttentionPool(self.config.hidden_size)
            self._init_weights(pool)
            pool_list.append(pool)
        self.pool_list = nn.ModuleList(pool_list)
        self.layernorm = nn.LayerNorm(self.config.hidden_size)
        fc_list = []
        for _ in range(self.n_target):
            fc = nn.Linear(self.config.hidden_size, 1)
            self._init_weights(fc)
            fc_list.append(fc)
        self.fc_list = nn.ModuleList(fc_list)

        self._re_init_layers(self.cfg.layer_reinitialize_n)
        if 0 <= self.cfg.freeze_n_layers:
            freeze(self.model.embeddings)
            freeze(
                self.model.encoder.layer[: self.cfg.freeze_n_layers]
            )  

    def _re_init_layers(
        self, 
        n_layers):
        if n_layers >= 1:
            for layer in self.model.encoder.layer[-n_layers:]:
                if hasattr(layer, "modules"):
                    for module in layer.modules():
                        for name, child in module.named_children():
                            init_type_name = self._init_weights(child)
                            if init_type_name is not None:
                                print(
                                    f"{name} is re-initialized, type: {init_type_name}, {module.__class__}"
                                )

    def _init_weights(
        self, 
        module
        ):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
            return "nn.Linear"
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            return "nn.Embedding"
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            return "nn.LayerNorm"
        return None

    def get_inputs_length(
        self, 
        inputs
        ):
        batch_size, token_length = inputs["attention_mask"].shape
        return token_length

    def split_inputs(
        self, 
        inputs, 
        i, 
        j):
        ret = {}
        for k, v in inputs.items():
            ret[k] = v[:, i:j]
        return ret

    def add_cls_token(
        self, 
        inputs):
        batch_size = inputs["input_ids"].shape[0]
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key == "token_type_ids":
                to_insert_first = torch.zeros((batch_size, 1), dtype=torch.long)
            else:
                to_insert_first = torch.ones((batch_size, 1), dtype=torch.long)
            to_insert_first = to_insert_first.to(inputs[key].device)
            inputs[key] = torch.cat([to_insert_first, inputs[key]], dim=1)
        return inputs

    def last_hidden_states_each_token(
        self, 
        inputs
        ) :
        outputs = self.model(**inputs)
        last_hidden_state = outputs[0]
        return last_hidden_state 

    def feature(
        self, 
        inputs
        ):
        token_length = self.get_inputs_length(inputs)
        if token_length <= self.window_size:
            last_hidden_states = self.last_hidden_states_each_token(
                inputs
            ) 
            attention_mask = inputs["attention_mask"]
        else: 
            n_segments = (token_length - self.edge_len) // self.except_edge_len
            if (token_length - self.edge_len) % self.except_edge_len:
                n_segments += 1

            last_hidden_states_list = [] 
            attention_mask_list = []
            segmented_inputs = self.split_inputs(inputs, 0, self.window_size + 1)
            last_hidden_states = self.last_hidden_states_each_token(segmented_inputs)
            last_hidden_states_list.append(last_hidden_states)
            attention_mask_list.append(segmented_inputs["attention_mask"])
            segment_indices = [(0, self.window_size + 1)]
            for i in range(1, n_segments):
                start = i * self.except_edge_len
                end = min(start + self.window_size, token_length)
                segment_indices.append((start, end))
                segmented_inputs = self.add_cls_token(
                    self.split_inputs(inputs, start, end)
                )
                last_hidden_states = self.last_hidden_states_each_token(
                    segmented_inputs
                )
                last_hidden_states_list.append(
                    last_hidden_states[:, 1:, :]
                )  
                attention_mask_list.append(segmented_inputs["attention_mask"][:, 1:])

            last_hidden_states = last_hidden_states_list[0]
            attention_mask = attention_mask_list[0]
            for seg, ((i1, j1), (i2, j2)) in enumerate(
                zip(segment_indices[:-1], segment_indices[1:]), start=1
            ):
                s = j1 - i2  
                t = j2 - i2 
                last_hidden_states = torch.cat(
                    [
                        last_hidden_states[:, 0 : (i2 + (s - s // 2)), :],
                        last_hidden_states_list[seg][:, (s // 2) : t, :],
                    ],
                    dim=1,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask[:, 0 : (i2 + (s - s // 2))],
                        attention_mask_list[seg][:, (s // 2) : t],
                    ],
                    dim=1,
                )

        features = []
        for i in range(self.n_target):
            feature = self.pool_list[i](last_hidden_states, attention_mask)
            features.append(feature)
        features = torch.stack(
            features, dim=1
        ) 
        return features  

    def forward(
        self, 
        inputs):
        features = self.layernorm(self.feature(inputs))
        outputs = []
        for i in range(self.n_target):
            if self.dropouts:
                output = sum(
                    [
                        self.fc_list[i](dropout(features[:, i, :]))
                        for dropout in self.dropouts
                    ]
                ) / len(self.dropouts)
            else:
                output = self.fc_list[i](features[:, i, :])
            outputs.append(output)
        output = torch.cat(outputs, dim=1) 
        return output 

