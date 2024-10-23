import torch
import torch.nn as nn
import torch.optim as optim
import re

# Tokenize Java code
def tokenize_java_code(java_code):
    tokens = re.findall(r'\b\w+\b|[^\w\s]', java_code)
    return tokens

# Encode tokenized Java code
def encode_tokenized_java_code(tokenized_java_code, vocab):
    encoded_java_code = [vocab.get(token, vocab['<UNK>']) for token in tokenized_java_code]
    return encoded_java_code

# Create vocabulary
def create_vocab(java_code_snippets):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for java_code in java_code_snippets:
        tokens = tokenize_java_code(java_code)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

# Memory-Augmented Neural Network (E2EMN)
class E2EMN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, memory_size):
        super(E2EMN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.memory = nn.Parameter(torch.randn(memory_size, embed_size))
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.code_parts = []  # Internal list to store code parts

    def forward(self, input_seq):
        # Concatenate stored code parts with the new input
        concatenated_code = ' '.join(self.code_parts) + ' ' + ' '.join(input_seq)
        tokenized_input = tokenize_java_code(concatenated_code)
        encoded_input = encode_tokenized_java_code(tokenized_input, vocab)

        # Convert encoded input to tensor and ensure correct dimensions
        embedded_input = self.embedding(torch.tensor(encoded_input).unsqueeze(0))
        memory_read = self.read_memory(embedded_input)
        lstm_input = embedded_input + memory_read
        lstm_output, _ = self.lstm(lstm_input)
        output = self.fc(lstm_output)

        return output

    def read_memory(self, input_embedding):
        # Simple memory read mechanism
        memory_read = torch.mean(self.memory, dim=0).unsqueeze(0).expand(input_embedding.size(0), -1, -1)
        return memory_read

    def add_code_part(self, code_part):
        # Add new code part to internal list
        self.code_parts.append(code_part)

    def generate_unit_test(self):
        # Generate unit test based on stored code parts
        concatenated_code = ' '.join(self.code_parts)
        tokenized_code = tokenize_java_code(concatenated_code)
        encoded_code = encode_tokenized_java_code(tokenized_code, vocab)
        input_seq = torch.tensor(encoded_code).unsqueeze(0)
        output = self.forward(input_seq)
        unit_test_case = convert_output_to_string(output, vocab)
        return unit_test_case

# Convert output to string
def convert_output_to_string(output, vocab):
    output_tokens = []
    output = output.argmax(dim=2)
    for seq in output:
        for token_idx in seq:
            token = [token for token, idx in vocab.items() if idx == token_idx.item()]
            if token:
                output_tokens.append(token[0])
    unit_test_case = ' '.join(output_tokens)
    return unit_test_case

# Sample Java code snippets and unit tests for vocabulary creation
java_code_snippets = [
    "public int add(int a, int b) { return a + b; }",
    "public boolean isEven(int number) { return number % 2 == 0; }",
    "public int factorial(int n) { if (n == 0) return 1; else return n * factorial(n - 1); }",
    "public int max(int a, int b) { return a > b ? a : b; }"
]

# Create vocabulary
vocab = create_vocab(java_code_snippets)
model = E2EMN(vocab_size=len(vocab), embed_size=64, hidden_size=128, memory_size=100)

# Code Assembler
assembler = model

# Example usage
print("Enter Java code parts to assemble and generate unit test:")
while True:
    code_part = input("Enter Java code part (e.g., method signature, variable declaration, etc.): ")
    assembler.add_code_part(code_part)
    response = input("Add another code part? (yes/no): ")
    if response.lower() != "yes":
        break

unit_test_case = assembler.generate_unit_test()
print("Generated Unit Test:")
print(unit_test_case)
