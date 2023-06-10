import torch
from torch import nn
from translation_dataset import get_english_german_translation_dataset
from torch.utils.data import DataLoader


class Transformer(nn.Module):
    def __init__(self, context_size, source_vocab_size, target_vocab_size, model_dim):
        super().__init__()
        self.positional_encoding = positional_encoding(context_size=context_size, embedding_dim=model_dim)
        self.src_embedding_layer = nn.Embedding(num_embeddings=source_vocab_size, embedding_dim=model_dim)
        self.tgt_embedding_layer = nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=model_dim)
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.linear = nn.Linear(in_features=model_dim, out_features=target_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def encode_source(self, src_sequence):
        src_embedded = self.src_embedding_layer(src_sequence) + self.positional_encoding
        src_encoding = self.encoder(src_embedded)
        return src_encoding

    def forward(self, src_encoding, tgt_sequence):
        tgt_embedded = self.tgt_embedding_layer(tgt_sequence) + self.positional_encoding
        tgt_decoded = self.decoder(src_encoding, tgt_embedded)

        output = self.linear(tgt_decoded)
        prb_next_token = self.softmax(output)

        return prb_next_token


if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    context_size = 64

    # Load the dataset
    dataset = get_english_german_translation_dataset(context_size=context_size)

    # Create a data loader for batching
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the model
    transformer = Transformer(source_vocab_size=dataset.source_vocab_size, target_vocab_size=dataset.target_vocab_size,
                              model_dim=16)
    transformer.eval()

    # Iterate over batches of data
    print("Batched Data:")
    for batch in data_loader:
        input_batch, target_batch = batch
        src_encoding = transformer.encode_source(input_batch)
        print(f"Input Batch: {input_batch.shape}")
        print(f"Output Batch: {transformer(input_batch).shape}")
        break
