# Decoder-Only Transformer Implementation

A PyTorch Lightning implementation of a decoder-only transformer model for sequence generation. This project demonstrates the core concepts of transformer architecture using a simple vocabulary to showcase the model's ability to learn and generate meaningful sequences.

## Architecture Overview

The model implements a decoder-only transformer with the following key components:

![Untitled diagram-2025-02-18-035623](https://github.com/user-attachments/assets/1aaa185b-e3ae-463e-a920-e79e93debae5)

- **Word Embeddings**: Converts token IDs into dense vector representations
- **Positional Encoding**: Adds position information to the embeddings using sinusoidal encoding
- **Self-Attention**: Implements scaled dot-product attention with causal masking
- **Residual Connections**: Helps maintain gradient flow and combine information
- **Fully Connected Layer**: Projects attention outputs to vocabulary space

## Features

- Implementation of core transformer components from scratch
- PyTorch Lightning integration for clean training loop organization
- Causal masking for autoregressive generation
- Simple vocabulary system for demonstration purposes
- Position encoding using sinusoidal functions

## Requirements

```
torch
pytorch-lightning
```

## Installation

```bash
pip install torch lightning
```

## Usage

The model can be used for simple sequence generation tasks. Here's a basic example:

```python
# Initialize the model
model = DecoderOnlyTransformer(
    num_tokens=len(token_to_id), 
    d_model=2, 
    max_len=6
)

# Prepare input
model_input = torch.tensor([token_to_id["What"],
                          token_to_id["is"],
                          token_to_id["Machine Learing"],
                          token_to_id["<EOS>"]])

# Generate sequence
predictions = model(model_input)
```

## Model Components

### Position Encoding

- Implements sinusoidal position encoding
- Supports sequences up to specified maximum length
- Uses alternating sine and cosine functions

### Attention Mechanism

- Scaled dot-product attention
- Query, Key, and Value linear projections
- Causal masking for autoregressive generation

### Training

The model is trained using:
- CrossEntropyLoss as the loss function
- Adam optimizer
- PyTorch Lightning Trainer for training loop management

## Example Output

The model can generate sequences like:

Input: "What is Machine Learning <EOS>"
Output: "Awesome <EOS>"

Input: "Machine Learning is What <EOS>"
Output: "Awesome <EOS>"

## Limitations

- Small vocabulary size for demonstration purposes
- Fixed embedding dimension (d_model=2)
- Simple architecture compared to full-scale transformer models

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.
