"""Categorical distribution/reproducibility checks without either framework."""
import numpy as np
from quest_repro.sampling import sample_top_k


def test_empirical_distribution_and_top_k_exclusion():
    logits=np.tile([10000.,10001.,10002.,-10000.],(30000,1))
    sampled=sample_top_k(logits,3,np.random.default_rng(0))
    expected=np.exp(np.array([0.,1.,2.]));expected/=expected.sum()
    np.testing.assert_allclose(np.bincount(sampled,minlength=4)[:3]/len(sampled),expected,atol=.012)
    assert not np.any(sampled==3)


def test_fixed_rng_replay_and_greedy_limit():
    logits=np.tile([.2,.7,-.5,1.3],(100,1))
    np.testing.assert_array_equal(sample_top_k(logits,3,np.random.default_rng(9)),sample_top_k(logits,3,np.random.default_rng(9)))
    assert np.all(sample_top_k(logits,1,np.random.default_rng(5))==3)


def test_reject_nonfinite_and_invalid_sampling_parameters():
    for values,k,t in [(np.array([[np.nan,0.]]),1,1.),(np.array([[np.inf,0.]]),1,1.),(np.zeros((1,3)),4,1.),(np.zeros((1,3)),2,0.)]:
        try:sample_top_k(values,k,np.random.default_rng(0),t)
        except ValueError:pass
        else:raise AssertionError('Invalid sampling configuration accepted')
