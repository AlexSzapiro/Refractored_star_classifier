import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load the dataset 
data_path = '/Users/as274094/GitHub/psf_dataset_generation/output/'
test_dataset = np.load(data_path + 'test_Euclid_res_10_TestStars_id_001GT_100_bins.npy', allow_pickle=True)[()]
train_dataset = np.load(data_path + 'train_Euclid_res_50_TrainStars_id_001GT_100_bins.npy', allow_pickle=True)[()]
output_path = '/Users/as274094/GitHub/Refractored_star_classifier/tensorflow_version/'
def SEDlisttoC(SED_list):
    sed_array = np.array(SED_list)
    return sed_array*0.5 + 1.5

train_stars = train_dataset['stars']
test_stars = test_dataset['stars']

y_train = SEDlisttoC(train_dataset['SED_ids'])
y_test = SEDlisttoC(test_dataset['SED_ids'])

PCA_components = 24
# Perform PCA on all the images
pca = PCA(n_components= PCA_components)
train_and_test_stars = np.concatenate((train_stars, test_stars), axis = 0)
pca.fit(train_and_test_stars.reshape(-1, 1024))
x_train = pca.transform(train_stars.reshape(-1, 1024))
x_test = pca.transform(test_stars.reshape(-1, 1024))
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 20000) # Reserve 20,000 stars for validation

PCA_dataset = {
    'train_stars_pca' : x_train,
    'validation_stars_pca' : x_val,
    'test_stars_pca' : x_test,
    'train_C' : y_train,
    'validation_C' : y_val,
    'test_C' : y_test,
    'test_SEDs' : test_dataset['SED_ids']
}

np.save(
    output_path + 'PCA_dataset1.npy',
    PCA_dataset,
    allow_pickle=True
)