# GPT-2 Model Reproduction

A PyTorch implementation of GPT-2 (Generative Pre-trained Transformer 2) from scratch, based on the original paper "Language Models are Unsupervised Multitask Learners".

## Overview

This project implements GPT-2, a state-of-the-art language model that uses the Transformer architecture to generate high-quality text. The implementation includes:

- **Transformer Architecture**: Multi-head self-attention mechanisms and feedforward networks
- **GPT Configuration**: Customizable model parameters (context window, hidden dimensions, number of layers, etc.)
- **Training Pipeline**: Complete training loop with batch processing
- **Text Generation**: Generate coherent text based on given prompts

## Project Structure

```
Transformer/
├── gpt2.ipynb       # Main implementation notebook
└── README.md        # This file
```

## Features

### 1. **Model Architecture**
- **Context Window (block_size)**: 512 tokens - the maximum sequence length the model can see at once
- **Batch Size**: 12 - processes multiple sequences in parallel for efficient training
- **Transformer Depth (n_layer)**: 12 layers - matching GPT-2 Small configuration
- **Hidden Dimension (n_embd)**: 768 - embedding and hidden layer dimensions
- **Attention Heads**: 12 - parallel attention mechanisms

### 2. **Key Components**
- **Token Embeddings**: Convert discrete tokens to continuous vectors
- **Positional Embeddings**: Encode position information in the sequence
- **Multi-Head Self-Attention**: Allow the model to attend to different representation subspaces
- **Feedforward Networks**: Non-linear transformations in each layer
- **Layer Normalization**: Stabilize training by normalizing activations

### 3. **Training Features**
- **PyTorch DataLoader**: Efficient batching and shuffling of training data
- **Configurable Parameters**: Easy adjustment of model size and training hyperparameters
- **Reproducible Results**: Fixed random seeds for consistent experiments

## Requirements

- Python 3.8+
- PyTorch
- Other dependencies listed in the notebook

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/gpt2-reproduction.git
cd gpt2-reproduction
```

2. Install dependencies:
```bash
pip install torch
```

## Usage

Open the Jupyter notebook in your development environment:

```bash
jupyter notebook Transformer/gpt2.ipynb
```

Run the cells in order to:
1. Import required libraries
2. Configure the GPT model parameters
3. Define the model architecture
4. Prepare training data
5. Train the model
6. Generate text

## Model Configuration

The `GPTConfig` dataclass allows you to customize model parameters:

```python
@dataclass
class GPTConfig:
    block_size: int = 512      # Context window size
    batch_size: int = 12       # Training batch size
    n_layer: int = 12          # Number of transformer layers
    n_embd: int = 768          # Embedding dimension
    n_head: int = 12           # Number of attention heads
    learning_rate: float = 3e-4 # Learning rate
    max_epochs: int = 10        # Maximum training epochs
```

## Implementation Details

### Transformer Block
Each transformer block consists of:
1. **Layer Normalization** → **Multi-Head Self-Attention** → **Residual Connection**
2. **Layer Normalization** → **Feedforward Network** → **Residual Connection**

### Attention Mechanism
Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

### Training Process
The model is trained using language modeling loss (next token prediction). The training loop:
1. Samples batches of sequences from the dataset
2. Computes forward pass through the model
3. Calculates cross-entropy loss
4. Performs backward pass and optimization step

## Performance

The implemented GPT-2 Small configuration achieves:
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- ~125 million parameters

## Future Improvements

- [ ] Add distributed training support (data parallelism)
- [ ] Implement gradient accumulation for larger effective batch sizes
- [ ] Add validation and evaluation metrics
- [ ] Optimize inference with generation strategies (beam search, top-k sampling)
- [ ] Add support for loading pre-trained weights
- [ ] Implement checkpoint saving and resuming

## References

- Original Paper: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- OpenAI GPT-2 Repository: https://github.com/openai/gpt-2
- Transformer Paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Created as an educational implementation for understanding the GPT-2 architecture and training process.

---

**Note**: This is a simplified implementation for educational purposes. For production use, consider using the official Hugging Face Transformers library.
