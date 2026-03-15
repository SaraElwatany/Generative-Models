# Generative-Models

Time Embedding: Enables temporal dynamics.

Skip Connections: Integrates fine-grained spatial details into the decoder via concatenation.

Flexible Architecture: Allows customization via model_config for varying depths, resolutions, and feature richness.

Normalization and Activation: GroupNorm ensures stable training, while SiLU activation improves non-linearity.

Output Consistency: Ensures the output image retains the original spatial dimensions and channel count.
