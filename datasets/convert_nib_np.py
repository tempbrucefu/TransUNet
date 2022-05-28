
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt

data_root= 'data/synapse_labels'
save_root= 'data/Synapse/labels_npz'

imgs = [f for f in os.listdir(data_root) if f.endswith(".nii.gz")]
for f in imgs:
    img = nib.load(os.path.join(data_root,f))
    a = np.array(img.dataobj)
    # a1= a[:,:,0]
    #imgplot = plt.imshow(a1)
    np.savez(os.path.join(save_root,f[:-7]),a)