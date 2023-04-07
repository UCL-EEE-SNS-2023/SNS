import os
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ydata_synthetic.synthesizers.timeseries import TimeGAN

from ydata_synthetic.synthesizers import ModelParameters

# Specific to TimeGANs

seq_len = 24        # Timesteps
n_seq = 6          # Features

# Hidden units for generator (GRU & LSTM).
# Also decides output_units for generator
hidden_dim = 24

gamma = 1           # Used for discriminator loss

noise_dim = 32      # Used by generator as a starter dimension
dim = 128           # UNUSED
batch_size = 128

learning_rate = 5e-3
beta_1 = 0          # UNUSED
beta_2 = 1          # UNUSED
data_dim = 28       # UNUSED

# batch_size, lr, beta_1, beta_2, noise_dim, data_dim, layers_dim
gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           noise_dim=noise_dim,
                           layers_dim=dim)

from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading

file_path = "./new_data.csv"
energy_df = pd.read_csv(file_path)

try:
    energy_df = energy_df.set_index('Date').sort_index()
except:
    energy_df = energy_df

# Data transformations to be applied prior to be used with the synthesizer model
energy_data = real_data_loading(energy_df.values, seq_len=seq_len)

print(len(energy_data), energy_data[0].shape)

# if path.exists('synth_energy.pkl'):
#     synth = TimeGAN.load('synth_energy.pkl')
# else:
synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=1)
synth.train(energy_data, train_steps=1000)
synth.save('synth_energy.pkl')


synth_data = synth.sample(len(energy_data))
print(synth_data.shape)

cols = ["Open","High","Low","Close","Volume","vader_sentiment"
]

# Plotting some generated samples. Both Synthetic and Original data are still standardized with values between [0, 1]
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes=axes.flatten()

time = list(range(1,1000))
obs = np.random.randint(len(energy_data))

for j, col in enumerate(cols):
    df = pd.DataFrame({'Real': energy_data[obs][:, j],
                   'Synthetic': synth_data[obs][:, j]})
    df.plot(ax=axes[j],
            title = col,
            secondary_y='Synthetic data', style=['-', '--'])
fig.tight_layout()

if not os.path.exists("./img"):
    os.makedirs("./img")
plt.savefig('img/comparison_gan_outputs.png', dpi=200)


# Scatter plots for PCA and t-SNE methods

import matplotlib.gridspec as gridspec

fig = plt.figure(constrained_layout=True, figsize=(20, 10))
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

ax = fig.add_subplot(spec[0,0])
ax.set_title('PCA results',
             fontsize=20,
             color='red',
             pad=10)

# PCA scatter plot
plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:, 1].values,
            c='black', alpha=0.2, label='Original')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

plt.scatter(pca_synth.iloc[:, 0], pca_synth.iloc[:, 1],
            c='red', alpha=0.2, label='Synthetic')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

ax.legend()

ax2 = fig.add_subplot(spec[0,1])
ax2.set_title('TSNE results',
              fontsize=20,
              color='red',
              pad=10)

# t-SNE scatter plot
plt.scatter(tsne_results.iloc[:7000, 0].values, tsne_results.iloc[:7000, 1].values,
            c='black', alpha=0.2, label='Original')
plt.scatter(tsne_results.iloc[7000:, 0], tsne_results.iloc[7000:, 1],
            c='red', alpha=0.2, label='Synthetic')

ax2.legend()

fig.suptitle('Validating synthetic vs real data diversity and distributions',
             fontsize=16,
             color='grey')

if not os.path.exists("./img"):
    os.makedirs("./img")
plt.savefig('img/synthetic_vs_real_data_diversity_and_distributions.png', dpi=200)