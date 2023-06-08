import torch
from torch import nn
from translation_dataset import get_english_german_translation_dataset
from torch.utils.data import DataLoader


class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, embedding_size, context_size):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=source_vocab_size, embedding_dim=embedding_size)
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.linear = nn.Linear(in_features=embedding_size * context_size, out_features=target_vocab_size)
        self.softmax = nn.Softmax(dim=1)

        self.src_encoding = None
        self.target_tokens = torch.empty(context_size)
        target_idx = 0
        self.target_tokens[target_idx] = bos_token_id

    def forward(self, src_tokens=None):

        if not self.src_encoding:
            src_embedded = self.embedding_layer(src_tokens)
            self.src_encoding = self.encoder(src_embedded)

        x = self.decoder(self.src_encoding, self.target_tokens)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        target_probability = self.softmax(x)
        self.target_tokens.append(torch.argmax(target_probability))

        return target_probability


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
                              embedding_size=16, context_size=context_size)
    transformer.eval()

    # Iterate over batches of data
    print("Batched Data:")
    for batch in data_loader:
        input_batch, target_batch = batch
        print(f"Input Batch: {input_batch.shape}")
        print(f"Output Batch: {transformer(input_batch).shape}")
        break
