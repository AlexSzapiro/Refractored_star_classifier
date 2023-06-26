# Removes SED information from PSF datasets

import numpy as np

dataset_path = '/Users/as274094/Documents/psf_dataset2/'
dataset_name = 'test_Euclid_res_20000_TestStars_id_002GT_100_bins.npy'
dataset = np.load(dataset_path + dataset_name, allow_pickle=True)[()]

dataset.pop('SEDs')
dataset.pop('SED_ids')

np.save(
        dataset_path + 'no_SED_' + dataset_name,
        dataset,
        allow_pickle=True
    )