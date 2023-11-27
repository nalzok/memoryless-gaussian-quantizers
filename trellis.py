import jax
import jax.numpy as jnp
import optax


def encode(samples, transition, emission, alphabet):
    S, C = transition.shape

    def f(rho, sample):
        next_rho = jnp.full((S,), jnp.inf)
        prev_state = jnp.empty((S,), dtype=int)
        prev_input = jnp.empty((S,), dtype=int)
        for state in range(S):
            for input_ in range(C):
                next_state = transition[state, input_]
                output = alphabet[emission[state, input_]]
                loss = rho[state] + (sample-output)**2
                pred = loss < next_rho[next_state]
                next_rho = next_rho.at[next_state].set(jax.lax.select(pred, loss, next_rho[next_state]))
                prev_state = prev_state.at[next_state].set(jax.lax.select(pred, state, prev_state[next_state]))
                prev_input = prev_input.at[next_state].set(jax.lax.select(pred, input_, prev_input[next_state]))

        return next_rho, (prev_state, prev_input)

    rho = jnp.full((S,), jnp.inf)
    rho = rho.at[0].set(0)
    rho, (prev_states, prev_inputs) = jax.lax.scan(f, rho, samples)

    def g(state, prev):
        prev_state, prev_input = prev
        return prev_state[state], prev_input[state]

    last_state = jnp.argmin(rho)
    _, encoded_rev = jax.lax.scan(g, last_state, (jnp.flipud(prev_states), jnp.flipud(prev_inputs)))
    encoded = jnp.flipud(encoded_rev)

    return encoded


def decode(encoded, transition, emission, alphabet):
    def f(state, input_):
        next_state = transition[state, input_]
        output = alphabet[emission[state, input_]]
        return next_state, output

    init_state = 0
    _, decoded = jax.lax.scan(f, init_state, encoded)

    return decoded
    

def evaluate(key, transition, emission, alphabet, n_samples):
    samples = jax.random.normal(key, (n_samples,))
    encoded = encode(samples, transition, emission, alphabet)
    decoded = decode(encoded, transition, emission, alphabet)
    residual = samples - decoded
    mse = jnp.mean(residual**2)

    bincount = jnp.bincount(encoded, length=4)
    dist = bincount / jnp.sum(bincount)
    entropy = -jnp.sum(dist * jnp.log2(dist))

    return mse, entropy


def train(key, transition, emission, alphabets, n_samples, learning_rate, n_steps):

    @jax.jit
    def train_step(key_step, ab, opt_state):
        ab = (ab - jnp.flipud(ab))/2    # enforce alphabet symmetry
        grad_fn = jax.value_and_grad(evaluate, argnums=3, has_aux=True)
        (mse, entropy), grads = grad_fn(key_step, transition, emission, ab, n_samples)
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

    for step in range(n_steps):
        key_step = jax.random.fold_in(key, step)
        mse, entropy, alphabets, opt_state = train_step(key_step, alphabets, opt_state)
        if step % (n_steps // 64) == 0:
            print(f"step #{step+1}, {mse.item() = :.4f}, {entropy.item() = :.4f}")

    return alphabets


def main(n_samples, learning_rate, n_steps):
    # TODO: If we use an eight-state trellis, each alphabet only contains one element. In this case,
    #       can we make it *completely* parallelizable by storing the states instead of inputs?

    # Ungerboeck's trellis (Fig. 3.5.) + rate-3 Lloyd-Max quantizer
    # MSE = 0.0880
    # transition = jnp.array([
    #     [0, 1, 0, 1],
    #     [2, 3, 2, 3],
    #     [0, 1, 0, 1],
    #     [2, 3, 2, 3],
    # ])
    # emission = jnp.array([
    #     [0, 2, 4, 6],
    #     [1, 3, 5, 7],
    #     [2, 0, 6, 4],
    #     [3, 1, 7, 5],
    # ])
    # alphabet_init = jnp.array([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152])

    # Quadrupled Output Alphabets (Chapter III.C & Table B.1) + rate-4 Lloyd-Max quantizer
    # MSE = 0.0821
    transition = jnp.array([
        [ 0,  2,  0,  2], [ 9, 11,  9, 11], [ 1,  3,  1,  3], [ 8, 10,  8, 10],
        [ 0,  2,  0,  2], [ 9, 11,  9, 11], [ 1,  3,  1,  3], [ 8, 10,  8, 10],
        [ 4,  6,  4,  6], [13, 15, 13, 15], [ 5,  7,  5,  7], [12, 14, 12, 14],
        [ 4,  6,  4,  6], [13, 15, 13, 15], [ 5,  7,  5,  7], [12, 14, 12, 14],
    ])
    emission = jnp.array([
        [ 4,  6,  8, 10], [ 5,  7,  9, 11], [ 2,  6, 14, 10], [ 5,  7,  9, 11],
        [ 0,  4, 12,  8], [ 7,  5, 11,  9], [ 6,  4, 10,  8], [ 7,  5, 11,  9],
        [ 4,  6,  8, 10], [ 5,  1,  9, 13], [ 4,  6,  8, 10], [ 5,  7,  9, 11],
        [ 6,  4, 10,  8], [ 7,  5, 11,  9], [ 6,  4, 10,  8], [ 7,  3, 11, 15],
    ])
    alphabet_init = jnp.array([
        -2.732, -2.068, -1.617, -1.256, -0.942, -0.657, -0.388, -0.128,
        0.128, 0.388, 0.657, 0.942, 1.256, 1.617, 2.068, 2.732,
    ])

    key = jax.random.PRNGKey(42)
    key_train, key_test = jax.random.split(key)

    alphabet = train(key_train, transition, emission, alphabet_init, n_samples, learning_rate, n_steps)
    print("Before", alphabet_init)
    print("After", alphabet)

    mse, entropy = evaluate(key_test, transition, emission, alphabet, 2**21)
    print(f"final: {mse.item() = :.4f}, {entropy.item() = :.4f}")



if __name__ == "__main__":
    n_samples = 2**10
    learning_rate = 1e-2
    n_steps = 2**10
    main(n_samples, learning_rate, n_steps)
