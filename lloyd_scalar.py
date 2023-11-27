import jax
import jax.numpy as jnp
import optax


def evaluate(key, codebook, n_samples):
    samples = jnp.abs(jax.random.normal(key, n_samples))
    differences = jnp.expand_dims(samples, -1) - codebook
    sq_distances = jnp.min(differences**2, axis=-1)
    mse = jnp.mean(sq_distances)
    return mse


def train(key, codebooks, n_samples, learning_rate, n_steps):

    @jax.jit
    def train_step(key_step, codebooks, opt_state):
        grad_fn = jax.value_and_grad(evaluate, argnums=1)
        mse, grads = grad_fn(key_step, codebooks, n_samples)
        updates, opt_state = gradient_transform.update(grads, opt_state, codebooks)
        codebooks = optax.apply_updates(codebooks, updates)

        return mse, codebooks, opt_state

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
    opt_state = gradient_transform.init(codebooks)

    for step in range(n_steps):
        key_step = jax.random.fold_in(key, step)
        mse, codebooks, opt_state = train_step(key_step, codebooks, opt_state)
        if step % (n_steps // 256) == 0:
            print(f"step #{step+1}, {mse.item() = :.4f}")

    return codebooks


def main(n_samples, learning_rate, n_steps):
    key = jax.random.PRNGKey(42)
    key_train, key_test = jax.random.split(key, 2)

    codebook = jnp.arange(8)/8
    codebook = train(key_train, codebook, n_samples, learning_rate, n_steps)
    print(codebook)

    mse = evaluate(key_test, codebook, 2**21)
    print(f"final: {mse.item() = :.4f}")



if __name__ == "__main__":
    n_samples = 2**10
    learning_rate = 1e-2
    n_steps = 2**10
    main(n_samples, learning_rate, n_steps)
