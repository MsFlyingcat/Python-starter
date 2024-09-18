import numpy as np
from scipy.stats import entropy

def kl_divergence(P, Q):
    # Ensure P and Q are valid probability distributions
    assert np.isclose(np.sum(P), 1), "P must sum to 1."
    assert np.isclose(np.sum(Q), 1), "Q must sum to 1."
    
    # Compute KL divergence using scipy's entropy function
    # Note: scipy's entropy function computes D_KL(P || Q)
    kl_div = entropy(P, Q)
    return kl_div

# 示例数据
P = np.array([0.1, 0.2, 0.7])
Q = np.array([0.2, 0.3, 0.5])

# 计算KL散度
kl_div = kl_divergence(P, Q)
print(f"KL divergence between P and Q: {kl_div}")