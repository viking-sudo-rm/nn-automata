import torch

from numpy import sqrt


def attention(queries, keys, values):
    """Compute scaled dot-product attention.

    Args:
        queries: A [batch_size, query_len] matrix of queries.
        keys: A [batch_size, seq_len, query_len] tensor of keys.
        values: A [batch_size, seq_len, value_len] tensor of values.
    """
    query_dim = queries.size(1)
    queries = queries.unsqueeze(dim=1)
    keys = keys.transpose(1, 2)
    relevance = torch.matmul(queries, keys)
    scaled_relevance = relevance / sqrt(query_dim)
    weights = torch.softmax(scaled_relevance, dim=2)
    return torch.matmul(weights, values).squeeze(dim=1)


def test_attention():
    queries = torch.zeros(10, 100)
    keys = torch.zeros(10, 32, 100)
    values = torch.zeros(10, 32, 64)
    attention_vector = attention(queries, keys, values)
    assert attention_vector.size() == torch.Size([10, 64])


if __name__ == "__main__":
    test_attention()
