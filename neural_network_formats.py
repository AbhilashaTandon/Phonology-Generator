import random


input_features = 719
latent_features = 20

for x in range(100):
    layers = gen_layers(input_features, latent_features)
    num_params = sum(a * b for a, b in zip(layers[:-1], layers[1:]))
    print(num_params, layers)
