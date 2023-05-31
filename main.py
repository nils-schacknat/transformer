import torch
from torch import nn
from dictionary import TokenDictionary

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


class Transformer(nn.Module):
    def __init__(self, dict_size, embedding_size, context_size):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=dict_size, embedding_dim=embedding_size)
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.linear = nn.Linear(in_features=embedding_size*context_size, out_features=dict_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = torch.tensor(x)
        x = self.embedding_layer(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.flatten()
        x = self.linear(x)

        if not self.training:
            x = self.softmax(x)

        return x


context_size = 16
input_sentence = "Ich bin ein <blank> Mensch"

dictionary = TokenDictionary()
dictionary.add_text(input_sentence)
input_tokens = dictionary.get_text_indices(input_sentence, target_length=context_size)

transformer = Transformer(dict_size=len(dictionary), embedding_size=16, context_size=context_size)
transformer.eval()
print(transformer(input_tokens))
