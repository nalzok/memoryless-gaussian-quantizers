import jax
import jax.numpy as jnp
import optax
import numpy as np


def e8p():
    codebook = np.load("data/e8p_abs.npy")
    return codebook


def evaluate(key, codebook, n_samples):
    samples = jax.random.normal(key, (n_samples, 8))
    differences = jnp.abs(jnp.expand_dims(samples, -2)) - codebook
    sq_distances = jnp.sum(differences**2, axis=-1)
    indices = jnp.argmin(sq_distances, axis=-1)
    min_sq_distances = sq_distances[jnp.arange(indices.shape[-1]), indices]
    mse = jnp.mean(min_sq_distances)
    bincount = jnp.bincount(indices, length=1<<8)
    dist = bincount / jnp.sum(bincount)
    entropy = -jnp.sum(dist * jnp.log2(dist))
    cb_norm = jnp.linalg.norm(codebook, axis=-1)
    min_norm = jnp.min(cb_norm)
    max_norm = jnp.max(cb_norm)
    return mse, (entropy, min_norm, max_norm)


def train(key, codebook, n_samples, learning_rate, n_steps):

    @jax.jit
    def train_step(key_step, codebook, opt_state):
        grad_fn = jax.value_and_grad(evaluate, argnums=1, has_aux=True)
        (mse, (entropy, min_norm, max_norm)), grads = grad_fn(key_step, codebook, n_samples)
        updates, opt_state = gradient_transform.update(grads, opt_state, codebook)
        codebook = optax.apply_updates(codebook, updates)

        return (mse, entropy, min_norm, max_norm), codebook, opt_state

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
    opt_state = gradient_transform.init(codebook)

    for step in range(n_steps):
        key_step = jax.random.fold_in(key, step)
        (mse, entropy, min_norm, max_norm), codebook, opt_state = train_step(key_step, codebook, opt_state)
        if step % (n_steps // 256) == 0:
            mse, entropy, min_norm, max_norm = mse.item(), entropy.item(), min_norm.item(), max_norm.item()
            print(f"step #{step+1}, {mse = :.4f}, {entropy = :.2f}, {min_norm = :.2f}, {max_norm = :.2f}")

    return codebook


def main(n_samples, learning_rate, n_steps):
    codebook = e8p()
    assert codebook.shape == (256, 8), codebook.shape

    key = jax.random.PRNGKey(42)
    key_train, key_test = jax.random.split(key)
    codebook = train(key_train, codebook, n_samples, learning_rate, n_steps)

    mse, (entropy, min_norm, max_norm) = evaluate(key_test, codebook, 2**21)
    mse, entropy, min_norm, max_norm = mse.item(), entropy.item(), min_norm.item(), max_norm.item()
    print(f"final: {mse = :.4f}, {entropy = :.2f}, {min_norm = :.2f}, {max_norm = :.2f}")



if __name__ == "__main__":
    n_samples = 2**10
    learning_rate = 1e-2
    n_steps = 2**10
    main(n_samples, learning_rate, n_steps)
