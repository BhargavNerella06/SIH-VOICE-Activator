import torch
import torch.nn as nn
import torch.nn.functional as F


def log_tensor(name, x, trace=None, print_values=False, max_values=8):
    """Print shape (and optional values) and optionally store a CPU copy."""
    # Print the tensor name and shape so data flow is visible.
    print(f"{name:<42} shape={tuple(x.shape)}")

    # Print a short value snippet to make internals less abstract.
    if print_values:
        flat = x.detach().reshape(-1)
        snippet = flat[:max_values].cpu().tolist()
        rounded = [round(float(v), 5) for v in snippet]
        print(f"{'':<42} first_values={rounded}")

    # Keep a copy so we can print a final summary after forward() ends.
    if trace is not None:
        trace[name] = x.detach().cpu()


class Encoder(nn.Module):
    """1-D analysis filterbank: waveform -> latent mixture representation."""

    def __init__(self, num_filters, kernel_size, stride):
        super().__init__()
        # Learned analysis filters U in Conv-TasNet papers.
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(self, wav, trace=None, verbose=False):
        # Input wav: [B, T]
        if verbose:
            log_tensor("input.waveform", wav, trace, print_values=True)

        # Add a channel axis expected by Conv1d: [B, T] -> [B, 1, T]
        x = wav.unsqueeze(1)
        if verbose:
            log_tensor("encoder.unsqueeze", x, trace)

        # Apply learned filterbank and ReLU to get non-negative activations.
        # Shape: [B, 1, T] -> [B, N, K]
        mixture_w = F.relu(self.conv1d(x))
        if verbose:
            log_tensor("encoder.mixture_w", mixture_w, trace, print_values=True)

        return mixture_w


class TemporalBlock(nn.Module):
    """One TCN block with dilation, residual path, and skip path."""

    def __init__(self, bottleneck_channels, hidden_channels, kernel_size, dilation):
        super().__init__()
        # Keep output length same as input length for odd kernel_size.
        padding = (kernel_size - 1) * dilation // 2

        # 1x1 projection: mixes channels at each time index.
        self.in_conv1x1 = nn.Conv1d(bottleneck_channels, hidden_channels, kernel_size=1)

        # Non-linearity and normalization improve optimization stability.
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)

        # Depthwise dilated convolution captures long temporal context.
        # groups=hidden_channels -> one filter per channel (depthwise).
        self.depthwise_dilated = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=hidden_channels,
        )

        # Second non-linearity + normalization.
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)

        # Residual output returns to bottleneck channel count.
        self.residual_out = nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=1)

        # Skip output also projected to bottleneck channel count.
        self.skip_out = nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=1)

        # Save dilation only for readable debug logs.
        self.dilation = dilation

    def forward(self, x, block_name="", trace=None, verbose=False):
        # x shape: [B, Bn, K]
        if verbose:
            log_tensor(f"{block_name}.input", x, trace)

        # Channel projection: [B, Bn, K] -> [B, H, K]
        h = self.in_conv1x1(x)
        if verbose:
            log_tensor(f"{block_name}.in_conv1x1", h, trace)

        # Activation + norm.
        h = self.prelu1(h)
        h = self.norm1(h)
        if verbose:
            log_tensor(f"{block_name}.prelu_norm1", h, trace)

        # Dilated convolution (THIS IS WHERE DILATION IS APPLIED).
        h = self.depthwise_dilated(h)
        if verbose:
            log_tensor(
                f"{block_name}.depthwise_dilated(d={self.dilation})", h, trace
            )

        # Activation + norm again.
        h = self.prelu2(h)
        h = self.norm2(h)
        if verbose:
            log_tensor(f"{block_name}.prelu_norm2", h, trace)

        # Residual branch and skip branch from same hidden state.
        residual = self.residual_out(h)
        skip = self.skip_out(h)
        if verbose:
            log_tensor(f"{block_name}.residual", residual, trace)
            log_tensor(f"{block_name}.skip", skip, trace)

        # Residual connection (THIS IS WHERE RESIDUAL CONNECTION IS USED).
        out = x + residual
        if verbose:
            log_tensor(f"{block_name}.output_residual_add", out, trace)

        # Return updated stream and skip stream.
        return out, skip


class SeparatorTCN(nn.Module):
    """Stacked temporal blocks that predict separation masks."""

    def __init__(
        self,
        num_filters,
        bottleneck_channels,
        hidden_channels,
        kernel_size,
        num_blocks,
        num_repeats,
        num_sources,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.num_filters = num_filters

        # Normalize encoder output over channels.
        self.layer_norm = nn.GroupNorm(num_groups=1, num_channels=num_filters)

        # Compress/expand channels for TCN processing.
        self.bottleneck = nn.Conv1d(num_filters, bottleneck_channels, kernel_size=1)

        # Build repeated blocks with exponentially increasing dilation.
        blocks = []
        for r in range(num_repeats):
            for b in range(num_blocks):
                dilation = 2 ** b
                blocks.append(
                    TemporalBlock(
                        bottleneck_channels=bottleneck_channels,
                        hidden_channels=hidden_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        # Convert aggregated skip features to mask logits.
        self.prelu = nn.PReLU()
        self.mask_conv = nn.Conv1d(
            bottleneck_channels, num_sources * num_filters, kernel_size=1
        )

    def forward(self, mixture_w, trace=None, verbose=False):
        # mixture_w shape: [B, N, K]
        if verbose:
            log_tensor("separator.input", mixture_w, trace)

        # Normalize then map to bottleneck channels: [B, N, K] -> [B, Bn, K]
        x = self.layer_norm(mixture_w)
        x = self.bottleneck(x)
        if verbose:
            log_tensor("separator.bottleneck", x, trace)

        # Accumulate skip outputs from all blocks.
        skip_sum = None
        for i, block in enumerate(self.blocks):
            block_name = f"separator.block{i}"
            x, skip = block(x, block_name=block_name, trace=trace, verbose=verbose)

            # Skip connection accumulation (THIS IS WHERE SKIP CONNECTIONS ARE USED).
            if skip_sum is None:
                skip_sum = skip
            else:
                skip_sum = skip_sum + skip

            if verbose:
                log_tensor(f"separator.skip_sum_after_block{i}", skip_sum, trace)

        # Turn accumulated skip features into raw mask scores.
        mask_logits = self.mask_conv(self.prelu(skip_sum))
        if verbose:
            log_tensor("separator.mask_logits", mask_logits, trace)

        # Reshape to [B, C, N, K] where C is number of sources.
        bsz, _, frames = mask_logits.shape
        masks = torch.sigmoid(mask_logits.view(bsz, self.num_sources, self.num_filters, frames))
        if verbose:
            log_tensor("separator.masks(sigmoid)", masks, trace, print_values=True)

        return masks


class Decoder(nn.Module):
    """Synthesis filterbank: masked latent representation -> waveform."""

    def __init__(self, num_filters, kernel_size, stride):
        super().__init__()
        # ConvTranspose1d acts like overlap-add synthesis with learned bases.
        self.deconv1d = nn.ConvTranspose1d(
            in_channels=num_filters,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(self, source_w, trace=None, verbose=False):
        # source_w shape: [B, C, N, K]
        if verbose:
            log_tensor("decoder.input_source_w", source_w, trace)

        # Merge batch and source axes so we decode each source independently.
        bsz, num_src, num_filters, frames = source_w.shape
        x = source_w.view(bsz * num_src, num_filters, frames)
        if verbose:
            log_tensor("decoder.reshape_for_deconv", x, trace)

        # Decode latent features to waveforms: [B*C, N, K] -> [B*C, 1, T_hat]
        decoded = self.deconv1d(x)
        if verbose:
            log_tensor("decoder.deconv1d", decoded, trace)

        # Remove the singleton channel and restore [B, C, T_hat].
        est_sources = decoded.squeeze(1).view(bsz, num_src, -1)
        if verbose:
            log_tensor("decoder.estimated_sources", est_sources, trace, print_values=True)

        return est_sources


class ConvTasNetMini(nn.Module):
    """Minimal, educational Conv-TasNet implementation with shape tracing."""

    def __init__(
        self,
        num_sources=2,
        num_filters=16,
        kernel_size=8,
        bottleneck_channels=12,
        hidden_channels=24,
        tcn_kernel_size=3,
        num_blocks=3,
        num_repeats=1,
    ):
        super().__init__()

        # 50% overlap is common in Conv-TasNet.
        stride = kernel_size // 2

        # Build encoder, separator, decoder.
        self.encoder = Encoder(num_filters=num_filters, kernel_size=kernel_size, stride=stride)
        self.separator = SeparatorTCN(
            num_filters=num_filters,
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            kernel_size=tcn_kernel_size,
            num_blocks=num_blocks,
            num_repeats=num_repeats,
            num_sources=num_sources,
        )
        self.decoder = Decoder(num_filters=num_filters, kernel_size=kernel_size, stride=stride)

    def forward(self, wav, verbose=False, return_trace=False):
        # Dictionary to keep intermediate tensors for a final summary.
        trace = {}

        # 1) Encode waveform into latent mixture weights.
        mixture_w = self.encoder(wav, trace=trace, verbose=verbose)

        # 2) Predict masks for each source using TCN separator.
        masks = self.separator(mixture_w, trace=trace, verbose=verbose)

        # 3) Apply masks to encoded mixture (THIS IS WHERE MASKING HAPPENS).
        mixture_w_expanded = mixture_w.unsqueeze(1)
        if verbose:
            log_tensor("masking.mixture_w_expanded", mixture_w_expanded, trace)

        source_w = masks * mixture_w_expanded
        if verbose:
            log_tensor("masking.source_w = masks * mixture_w", source_w, trace, print_values=True)

        # 4) Decode each source back to time-domain waveform.
        est_sources = self.decoder(source_w, trace=trace, verbose=verbose)

        # Return both output and trace if caller wants internals.
        if return_trace:
            return est_sources, trace
        return est_sources



def print_flow_summary(trace):
    """Pretty print a compact, ordered summary of tensor flow."""
    print("\n" + "=" * 80)
    print("FORWARD PASS FLOW SUMMARY")
    print("=" * 80)
    for name, tensor in trace.items():
        print(f"{name:<42} -> {tuple(tensor.shape)}")



def print_intermediate_snippets(trace, keys, max_values=8):
    """Print a few values from selected intermediate tensors."""
    print("\n" + "=" * 80)
    print("INTERMEDIATE OUTPUT SNIPPETS")
    print("=" * 80)
    for key in keys:
        if key in trace:
            vals = trace[key].reshape(-1)[:max_values].tolist()
            vals = [round(float(v), 5) for v in vals]
            print(f"{key:<42} values={vals}")



def demo():
    """Run a tiny example: batch size = 1, short signal length."""
    # Deterministic demo so printed numbers are reproducible.
    torch.manual_seed(0)

    # Create model.
    model = ConvTasNetMini(
        num_sources=2,
        num_filters=16,
        kernel_size=8,
        bottleneck_channels=12,
        hidden_channels=24,
        tcn_kernel_size=3,
        num_blocks=3,
        num_repeats=1,
    )

    # Example input: B=1, T=32 (small signal, easy to inspect).
    # Shape: [1, 32]
    wav = torch.tensor(
        [[
            0.00, 0.20, 0.35, 0.40, 0.10, -0.10, -0.30, -0.40,
            -0.15, 0.05, 0.22, 0.38, 0.18, -0.05, -0.25, -0.35,
            -0.10, 0.12, 0.30, 0.36, 0.14, -0.08, -0.26, -0.33,
            -0.05, 0.16, 0.34, 0.39, 0.11, -0.12, -0.28, -0.36,
        ]],
        dtype=torch.float32,
    )

    # Forward pass with verbose logging and trace collection.
    with torch.no_grad():
        est_sources, trace = model(wav, verbose=True, return_trace=True)

    # Print final output shape.
    print("\nFinal estimated sources shape:", tuple(est_sources.shape))

    # Print a compact flow summary and selected value snippets.
    print_flow_summary(trace)
    print_intermediate_snippets(
        trace,
        keys=[
            "input.waveform",
            "encoder.mixture_w",
            "separator.mask_logits",
            "separator.masks(sigmoid)",
            "masking.source_w = masks * mixture_w",
            "decoder.estimated_sources",
        ],
    )


if __name__ == "__main__":
    demo()
