import torch
import torch.nn.functional as F
from torch import nn


class SharedLinear(nn.Module):
    def __init__(self, in_dim=26, out_dim=100):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)  # No bias

    def forward(self, x, transpose=False):
        if transpose:
            return torch.matmul(x, self.linear.weight.T)  # Use transposed weights
        else:
            return self.linear(x)


class GP_model(nn.Module):
    def __init__(
        self,
        tokens_size,
        directions_size,
        embedding_size,
        hidden_size,
        dropout,
        n_layers,
        output_size,
        layer_norm=False,
    ):
        super().__init__()
        self.token_proj = SharedLinear(tokens_size, embedding_size // 2)
        self.coord_proj = SharedLinear(directions_size, embedding_size // 2)

        self.embedding = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
        )

        self.rnn = nn.GRU(
            hidden_size, hidden_size, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )

        self.rnn_norm = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
        self.fc_proj = SharedLinear(hidden_size, hidden_size)
        self.readout = SharedLinear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

    @staticmethod
    def _as_batch_seq(x: torch.Tensor) -> torch.Tensor:
        # Accept (T, D) or (B, T, D). Normalize to (B, T, D).
        if x.dim() == 2:
            return x.unsqueeze(0)
        return x

    def _gru_unroll_with_gates(self, emb, initial_hidden=None, update_gate_masks=None):
        """
        Manual multi-layer GRU unroll that matches PyTorch's GRU math.
        Returns:
          top_out: (B,T,H)
          layer_acts: list of (B,T,H) length n_layers
          gate_dict: {'update','reset','candidate'} each list of (B,T,H)
        """
        B, T, H = emb.shape
        device = emb.device
        dtype = emb.dtype
        L = self.n_layers

        # ---- initial hidden -> list[h_l] where each is (B,H)
        if initial_hidden is None:
            h = [torch.zeros(B, H, device=device, dtype=dtype) for _ in range(L)]
        else:
            h0 = initial_hidden
            # GRU convention: (num_layers, B, H)
            if isinstance(h0, (tuple, list)):  # be tolerant; pick first element if someone passed (h,c)
                h0 = h0[0]
            if h0.dim() == 3:
                # unbind layers
                h = [h0[l] for l in range(L)]
            elif h0.dim() == 2:
                # treat as single-layer init; other layers zeros
                h = [h0] + [torch.zeros(B, H, device=device, dtype=dtype) for _ in range(L - 1)]
            else:
                raise ValueError("initial_hidden must be None, (L,B,H), or (B,H) for GRU.")

        # ---- normalize update_gate_masks to list length L (each mask broadcastable to (B,H))
        masks = None
        if update_gate_masks is not None:
            if isinstance(update_gate_masks, (list, tuple)):
                masks = list(update_gate_masks)
            elif torch.is_tensor(update_gate_masks):
                # allow (L,B,H) or (L,H)
                if update_gate_masks.dim() == 3:
                    masks = [update_gate_masks[l] for l in range(L)]
                elif update_gate_masks.dim() == 2:
                    masks = [update_gate_masks[l] for l in range(L)]
                else:
                    raise ValueError("update_gate_masks tensor must be (L,B,H) or (L,H).")
            else:
                raise ValueError("update_gate_masks must be list/tuple or tensor.")

            if len(masks) != L:
                raise ValueError(f"update_gate_masks length {len(masks)} != n_layers {L}")

        # ---- cache GRU params (avoid getattr in loops)
        W_ih = [getattr(self.rnn, f"weight_ih_l{l}") for l in range(L)]
        W_hh = [getattr(self.rnn, f"weight_hh_l{l}") for l in range(L)]
        b_ih = [getattr(self.rnn, f"bias_ih_l{l}") for l in range(L)]
        b_hh = [getattr(self.rnn, f"bias_hh_l{l}") for l in range(L)]

        # ---- preallocate outputs
        top_out = torch.empty(B, T, H, device=device, dtype=dtype)
        layer_acts = [torch.empty(B, T, H, device=device, dtype=dtype) for _ in range(L)]
        z_store = [torch.empty(B, T, H, device=device, dtype=dtype) for _ in range(L)]
        r_store = [torch.empty(B, T, H, device=device, dtype=dtype) for _ in range(L)]
        n_store = [torch.empty(B, T, H, device=device, dtype=dtype) for _ in range(L)]

        # dropout between stacked GRU layers (PyTorch applies between layers, not on last)
        p_drop = float(getattr(self.rnn, "dropout", 0.0))

        for t in range(T):
            x = emb[:, t, :]  # (B,H)

            for l in range(L):
                x_in = x if l == 0 else x  # x is already previous layer output after dropout (see below)
                h_prev = h[l]

                # i = W_ih x + b_ih ; hlin = W_hh h + b_hh
                i_lin = F.linear(x_in, W_ih[l], b_ih[l])
                h_lin = F.linear(h_prev, W_hh[l], b_hh[l])

                # PyTorch GRU gate order: reset, update, new  (r, z, n)
                i_r, i_z, i_n = i_lin.chunk(3, dim=1)
                h_r, h_z, h_n = h_lin.chunk(3, dim=1)

                r = torch.sigmoid(i_r + h_r)
                z = torch.sigmoid(i_z + h_z)
                n = torch.tanh(i_n + r * h_n)

                if masks is not None and masks[l] is not None:
                    # broadcast to (B,H) if needed
                    z = z * masks[l]

                h_new = n + z * (h_prev - n)  # same as (1-z)*n + z*h_prev
                h[l] = h_new

                layer_acts[l][:, t, :] = h_new
                z_store[l][:, t, :] = z
                r_store[l][:, t, :] = r
                n_store[l][:, t, :] = n

                # feed to next layer; apply inter-layer dropout in training
                x = h_new
                if self.training and p_drop > 0.0 and l < (L - 1):
                    x = F.dropout(x, p=p_drop, training=True)

            top_out[:, t, :] = h[-1]

        gate_dict = {
            "update":   [g.detach() for g in z_store],
            "reset":    [g.detach() for g in r_store],
            "candidate":[g.detach() for g in n_store],
        }
        layer_acts = [a.detach() for a in layer_acts]
        return top_out, layer_acts, gate_dict


    def forward(self, t, c, return_all_activations=False, initial_hidden=None, update_gate_masks=None):
        """
        t: token one-hot, (B,T,tokens_size) or (T,tokens_size)
        c: coords,       (B,T,directions_size) or (T,directions_size)
        """
        t = self._as_batch_seq(t)
        c = self._as_batch_seq(c)

        token_emb = self.token_proj(t)   # (B,T,E/2)
        coord_emb = self.coord_proj(c)   # (B,T,E/2)

        proj = torch.cat((token_emb, coord_emb), dim=2)  # (B,T,E)
        proj = self.dropout(proj)
        emb = self.embedding(proj)  # (B,T,H)

        if not return_all_activations:
            final_output, _ = self.rnn(emb, initial_hidden)
            final_output = self.rnn_norm(final_output)
            fc_out = F.relu(self.fc_proj(final_output))
            return self.readout(fc_out, transpose=True)


        final_output, layer_activations, gate_dict = self._gru_unroll_with_gates(
            emb, initial_hidden=initial_hidden, update_gate_masks=update_gate_masks
        )

        final_output = self.rnn_norm(final_output)
        fc_out = F.relu(self.fc_proj(final_output))
        output = self.readout(fc_out, transpose=True)
        return output, layer_activations, gate_dict