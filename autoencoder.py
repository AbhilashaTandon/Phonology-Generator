import torch

import numpy as np
from generation import gen_avg_inventory, gen_rand_inventory


class AE(torch.nn.Module):
    # features and compressed features
    def __init__(
        self,
        latent_features,
        layers,
        lr,
        batch_size,
        weight_decay,
        load_from="",
    ):
        super().__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.encoder = torch.nn.Sequential()

        self.decoder = torch.nn.Sequential()

        for x, y in zip(layers[:-1], layers[1:]):
            self.encoder.append(torch.nn.Linear(x, y))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(layers[-1], latent_features))

        self.decoder.append(torch.nn.Linear(latent_features, layers[-1]))

        layers = layers[::-1]
        for x, y in zip(layers[:-1], layers[1:]):
            self.decoder.append(torch.nn.ReLU())
            self.decoder.append(torch.nn.Linear(x, y))
        self.decoder.append(torch.nn.Sigmoid())

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.3, mode="min"
        )

        if load_from != "":
            checkpoint = torch.load(load_from)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, vec):
        decoded = self.decoder(vec)
        return decoded

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(
        self,
        train_data,
        test_data,
        phoneme_list,
        phoneme_to_id,
        loss_function,
        epochs,
        verbose=True,
        save_to="",
    ):
        for epoch in range(epochs):
            avg_loss = 0
            avg_test_loss = 0
            np.random.shuffle(train_data)
            num_batches = len(train_data) // self.batch_size
            test_batch_size = len(test_data) // num_batches
            for i in range(num_batches):
                avg_loss += self.train_batch(
                    train_data,
                    phoneme_list,
                    phoneme_to_id,
                    loss_function,
                    self.batch_size,
                    i,
                )

                with torch.no_grad():
                    avg_test_loss += self.train_batch(
                        test_data,
                        phoneme_list,
                        phoneme_to_id,
                        loss_function,
                        test_batch_size,
                        i,
                        train=False,
                    )

            avg_loss /= num_batches
            avg_test_loss /= num_batches
            if verbose:
                print("Epoch #%d\t%.6f\t%.6f" % (epoch, avg_loss, avg_test_loss))

            self.scheduler.step(avg_loss)

        if save_to != "":
            torch.save(
                {
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                },
                save_to,
            )
            print("MODEL SAVED")

        return avg_loss, avg_test_loss

    def train_batch(
        self,
        data,
        phoneme_list,
        phoneme_to_id,
        loss_function,
        batch_size,
        i,
        train=True,
    ):
        min_index = i * batch_size
        max_index = min((i + 1) * batch_size, len(data))
        batch = np.array(
            [
                data[x].get_vector(phoneme_list, phoneme_to_id)
                for x in range(min_index, max_index)
            ]
        )

        batch = torch.Tensor(batch).reshape(-1, len(phoneme_list))

        reconstructed = self(batch)

        mse_loss = loss_function(reconstructed, batch)

        loss = mse_loss

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return mse_loss.detach().item()

    def test(self, test_data, phoneme_list, phoneme_to_id, loss_function):
        with torch.no_grad():
            avg_loss = 0
            for idx, lang in enumerate(test_data):
                input = lang.get_vector(phoneme_list, phoneme_to_id)
                input = torch.Tensor(input.reshape(len(phoneme_list)))

                reconstructed = self(input)

                mse_loss = loss_function(reconstructed, input)

                avg_loss += mse_loss.detach().item()
            return avg_loss / len(test_data)

    def post_training(self, languages, phoneme_list, phoneme_to_id, latent_features):
        # get average encoded vector
        with torch.no_grad():
            enc_vector = []
            for i in range(len(languages)):
                inventory_vec = torch.Tensor(
                    languages[i].get_vector(phoneme_list, phoneme_to_id)
                )
                encoded = self.encode(inventory_vec).detach().numpy()
                enc_vector.append(encoded)

            enc_vector = np.array(enc_vector)

            mean = np.mean(enc_vector, axis=0)
            stdev = np.std(enc_vector, axis=0)

            for x in range(10):
                inventory = gen_rand_inventory(
                    self, mean, stdev, phoneme_list, latent_features
                )
                print(inventory)

            print("Average Inventory")
            print(gen_avg_inventory(self, mean, phoneme_list, latent_features))
