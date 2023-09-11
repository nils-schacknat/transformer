## Transformer implementation for English to German translation

This project is a university assignment, implementing the Transformer architecture as introduced in ‘[Attention is All You Need](https://arxiv.org/abs/1706.03762)’ by Vaswani et al. Utilizing cloud-based training, I have been able to replicate the reported results for English to German translation on their specified ‘base’ model.

## Methodology
- **Implementation**:
The implementation of the architecture is done in 'plain' PyTorch. This means all Transformer components are implemented using tensor arithmetic and basic structures such as forward layers, embedding layers and layer norm.
- **Cloud Computing**:
The training process was hosted on the [Lambda Labs](https://lambdalabs.com/) platform, utilizing an NVIDIA RTX A6000 GPU with 48GB of memory.

I also made use of [ChatGPT](https://chat.openai.com/) 3.5, primarily for code documentation, but also for some assistance in coding and writing.

## Experimental Setup

The parameter configuration used for the model mirrors the 'base' model described in the original paper. The training configuration remains the same as well, with minor variations:

- **Training Steps**: 50K (100K originally, limited for cost reasons, but sufficient for convergence).
- **Batch Size**: ~20K source and target tokens per batch each (~25K originally).
- **Single Precision Training**: The training is performed with single precision floats to compensate for the smaller GPU memory capacity (48GB vs 8×16GB).
- **Data Filtering**: The dataset underwent filtering to remove mismatched sequence pairs, improving training quality.

## Results

- **BLEU Score**: The model achieved a BLEU score of 25.9, confidently matching the reported value of 25.8, indicating a successful replication of the architecture.

- **Translation Examples**: Several translation examples are provided in [demo_notebooks/translation_demo.ipynb](demo_notebooks/translation_demo.ipynb), showcasing the model's performance on a variety of sentences.

You can download the latest training checkpoint, including the model weights and TensorBoard logs from [this link](https://heibox.uni-heidelberg.de/d/526fdc6dac434841bde1/). (For some time at least)

## Installation Instructions

To run the implementation locally, clone this repository and follow the steps below:

```bash
pip install -r requirements.txt            # Install the requirements
python main.py                             # Start the training
```
\
**Author**: Nils Schacknat\
**Date**: 11.09.2023