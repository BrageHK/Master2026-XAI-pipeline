import os, sys
os.environ['RESULTS_FOLDER'] = 'results/nnUNet'
os.environ['nnUNet_raw_data_base'] = 'nnunet_base'
os.environ['nnUNet_preprocessed'] = 'nnunet_base/nnUNet_preprocessed'
sys.path.insert(0, 'nnUNet')
sys.path.insert(0, '.')

# Import functions from captum_xai
from captum_xai import load_plans, load_fold_network, preprocess_case, generate_xai

print('Loading plans...')
plans = load_plans()

print('Loading fold 0 network...')
network = load_fold_network(0)

from collections import deque

#last = deque(network.modules(), maxlen=20)
# Print the model architecture to identify layer names
for name, module in network.named_modules():
    print(name, type(module))

print(network.conv_blocks_context[6][1].blocks[0].conv)

#print(last)

import sys
sys.exit(0)

device = next(network.parameters()).device
print(f'Network device: {device}')

print('Preprocessing 10001_1000001...')
data = preprocess_case('10001_1000001', plans)
print(f'Data shape: {data.shape}')

print('Generating XAI (saliency + occlusion)...')
sal, occ = generate_xai(
    network, data, plans,
    occlusion_window=(1, 8, 128, 128),
    occlusion_stride=(1, 8, 128, 128),
    perturbations_per_eval=1,
)
print(f'Saliency shape: {sal.shape}, range: [{sal.min():.4f}, {sal.max():.4f}]')
print(f'Occlusion shape: {occ.shape}, range: [{occ.min():.4f}, {occ.max():.4f}]')
print('SUCCESS')