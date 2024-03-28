from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from tqdm import trange


def quantize(permutation, alphabet, scales, samples):
    M, = alphabet.shape
    L = M.bit_length() - 1
    assert len(scales.shape) == len(samples.shape) == 1

    def f(rho, scale_sample):
        scale, sample = scale_sample
        states = jnp.arange(M)
        discarded = jnp.arange(1<<2)
        last_state = (discarded << (L-2)) | (states[..., jnp.newaxis] >> 2)
        output = alphabet[permutation[states[..., jnp.newaxis]]]
        loss = rho[last_state] + scale*(sample-output)**2

        next_rho = jnp.min(loss, axis=-1)
        optimal_discarded = jnp.argmin(loss, axis=-1)
        prev_state = (optimal_discarded << (L-2)) | (states >> 2)
        prev_input = states & 0b11

        return next_rho, (prev_state, prev_input)

    rho = jnp.full((M,), jnp.inf)
    rho = rho.at[0].set(0)
    rho, (prev_states, prev_inputs) = jax.lax.scan(f, rho, (scales, samples))

    def g(state, prev):
        prev_state, prev_input = prev
        return prev_state[state], prev_input[state]

    last_state = jnp.argmin(rho)
    zero, quantized_rev = jax.lax.scan(g, last_state, (jnp.flipud(prev_states), jnp.flipud(prev_inputs)))
    quantized = jnp.flipud(quantized_rev)

    return quantized


def dequantize(permutation, alphabet, quantized):
    M, = alphabet.shape

    def f(state, input_):
        next_state = (M-1) & ((state<<2) | input_)
        output = alphabet[permutation[next_state]]
        return next_state, output

    init_state = 0
    _, dequantized = jax.lax.scan(f, init_state, quantized)

    return dequantized
    

def evaluate(permutation, alphabet, scales, samples):
    quantized = quantize(permutation, alphabet, scales, samples)
    dequantized = dequantize(permutation, alphabet, quantized)
    residual = samples - dequantized
    mse = jnp.mean(residual**2)

    bincount = jnp.bincount(quantized, length=4)
    dist = bincount / jnp.sum(bincount)
    entropy = -jnp.sum(dist * jnp.log2(dist))

    return mse, entropy


def train(key, permutation, alphabets, block_size, learning_rate, n_steps):

    @jax.jit
    def train_step(key_step, ab, opt_state):
        ab = (ab - jnp.flipud(ab))/2    # enforce alphabet symmetry
        grad_fn = jax.value_and_grad(evaluate, argnums=1, has_aux=True)
        samples = jax.random.normal(key_step, (block_size,))
        scales = jnp.ones_like(samples)
        (mse, entropy), grads = grad_fn(permutation, ab, scales, samples)
        updates, opt_state = gradient_transform.update(grads, opt_state, ab)
        ab = optax.apply_updates(ab, updates)

        return mse, entropy, ab, opt_state

    scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=learning_rate,
            warmup_steps=n_steps // 256,
            decay_steps=n_steps)
    gradient_transform = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0),
    )
    opt_state = gradient_transform.init(alphabets)

    for step in (pbar := trange(n_steps)):
        key_step = jax.random.fold_in(key, step)
        mse, entropy, alphabets, opt_state = train_step(key_step, alphabets, opt_state)
        pbar.set_description(f"{mse.item() = :.4f}, {entropy.item() = :.4f}")

    return alphabets


def main(block_size, learning_rate, n_steps):
    L = 16
    M = 1<<L
    permutation = jax.random.permutation(jax.random.PRNGKey(42), M)
    alphabet = jsp.stats.norm.ppf((2*jnp.arange(M)+1)/2/M)
    print("Before", alphabet)

    key = jax.random.PRNGKey(42)
    key_train, key_test = jax.random.split(key)

    # alphabet = train(key_train, permutation, alphabet, block_size, learning_rate, n_steps)
    # print("After", alphabet)

    samples = jax.random.normal(key_test, (2**20//block_size, block_size))
    scales = jnp.ones(block_size)
    mse_all, entropy_all = jax.lax.map(partial(evaluate, permutation, alphabet, scales), samples)
    mse_mean = jnp.mean(mse_all).item()
    mse_std = jnp.std(mse_all).item()
    entropy_mean = jnp.mean(entropy_all).item()
    entropy_std = jnp.std(entropy_all).item()
    print(f"final: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")



if __name__ == "__main__":
    block_size = 2**10
    learning_rate = 1e-2
    n_steps = 2**10
    main(block_size, learning_rate, n_steps)
