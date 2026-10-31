"""Framework-independent stable top-k categorical sampling for deployment."""
import numpy as np


def sample_top_k(logits,top_k,rng,temperature=1.0):
    logits=np.asarray(logits,dtype=np.float64)
    if logits.ndim!=2 or not np.isfinite(logits).all():
        raise ValueError('Expected finite batch-by-vocabulary logits')
    if not 1<=top_k<=logits.shape[1] or not np.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid top-k or temperature')
    labels=np.argsort(-logits,axis=1,kind='stable')[:,:top_k]
    if top_k==1:return labels[:,0].astype(np.int32)
    values=np.take_along_axis(logits,labels,axis=1)/temperature
    probabilities=np.exp(values-values.max(axis=1,keepdims=True))
    probabilities/=probabilities.sum(axis=1,keepdims=True)
    # Inverse CDF uses one uniform per batch row. Same RNG+logits are repeatable.
    uniform=rng.random(len(logits))
    selected=(uniform[:,None]>np.cumsum(probabilities,axis=1)).sum(axis=1)
    return labels[np.arange(len(logits)),np.minimum(selected,top_k-1)].astype(np.int32)
