import re
import nltk
from typing import Optional, List


class TokenDictionary:
    def __init__(self):
        """
        TokenDictionary class to manage tokens and their indices.
        """
        self.token_dictionary = {"<pad>": 0, "<blank>": 1, "<eos>": 2}
        self.current_index = 3
        self.special_char_pattern = r"(.*)(<.*>)(.*)"

    def word_tokenize(self, word: str) -> List[str]:
        """
        Tokenizes a word, handling special characters enclosed in '<>'.

        The function uses recursion to tokenize the parts of the word enclosed in '<>'.
        It extracts the parts using regex and tokenizes each part separately.
        If a part is not enclosed in '<>', it is tokenized using nltk's word_tokenize function.

        Args:
            word (str): The word to tokenize.

        Returns:
            list[str]: The list of tokens.
        """
        match = re.search(self.special_char_pattern, word)
        if match:
            parts = [part for part in match.groups() if part]
            if len(parts) == 1:
                return parts
            else:
                token_list = []
                for word in parts:
                    token_list.extend(self.word_tokenize(word))
                return token_list

        else:
            return nltk.word_tokenize(word)

    def add_token(self, token: str) -> None:
        """
        Adds a token to the dictionary if it doesn't exist.

        Args:
            token (str): The token to add.
        """
        if token not in self.token_dictionary:
            self.token_dictionary[token] = self.current_index
            self.current_index += 1

    def add_text(self, text: str) -> None:
        """
        Adds tokens from the given text to the dictionary.

        Args:
            text (str): The text to process.
        """
        for word in text.split():
            for token in self.word_tokenize(word):
                self.add_token(token)

    def get_token_index(self, token: str) -> int:
        """
        Retrieves the index of a token from the dictionary.

        Args:
            token (str): The token to retrieve the index for.

        Returns:
            int: The index of the token in the dictionary, or None if not found.
        """
        return self.token_dictionary.get(token)

    def get_text_indices(self, text: str, target_length: Optional[int] = None) -> List[int]:
        """
        Retrieves the indices of tokens from the given text.

        Args:
            text (str): The text to process.
            target_length (Optional[int]): The desired length of the resulting index list.
                If provided, the index list will be padded with 0s (padding tokens) to match
                the target length. Default is None.

        Returns:
            List[int]: The list of token indices.
        """
        index_list = []

        for word in text.split():
            for token in self.word_tokenize(word):
                index_list.append(self.get_token_index(token))

        if target_length is not None:
            # Add padding tokens
            index_list.extend([0] * (target_length - len(index_list)))

        return index_list

    def __len__(self) -> int:
        """
        Returns the length of the token dictionary.

        Returns:
            int: The number of tokens in the dictionary.
        """
        return len(self.token_dictionary)

    @property
    def tokens(self) -> list[str]:
        """
        Property that returns a list of all the tokens (keys) in the dictionary.

        Returns:
            list[str]: A list of all the tokens in the dictionary.
        """
        return list(self.token_dictionary.keys())


if __name__ == "__main__":
    token_dict = TokenDictionary()

    # Add tokens from text
    token_dict.add_text("Hello, <name>! How are you doing? <eos>")

    # Add individual tokens
    token_dict.add_token("greeting")
    token_dict.add_token("you")

    # Get token index
    greeting_index = token_dict.get_token_index("greeting")
    name_index = token_dict.get_token_index("name")

    print("Tokens:")
    print(token_dict.tokens)
    print()

    print("Token indices for the text:")
    text = "Hello, <eos><name>! How are you you doing?"
    indices = token_dict.get_text_indices(text)
    print(indices)
    print()

    print("Token index for 'greeting':", greeting_index)
    print("Token index for 'name':", name_index)
    print()

    print("Token dictionary length:", len(token_dict))
