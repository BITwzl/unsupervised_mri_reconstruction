import random
import pathlib
import numpy as np
import h5py
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from utils import *
from fastmri.data import transforms as T
from mri_tools import *
from dataset.prob_mask import *

import os
import xml.etree.ElementTree as etree
def et_query(
    root: etree.Element,
    qlist,
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)

    
class CustomFastMRIDataSet(Dataset):
    def __init__(self, data_path, mask_configs, sample_rate):
        self.data_path = data_path
        self.sample_rate = sample_rate

        self.examples = []
        files = list(pathlib.Path(self.data_path).iterdir())
        if "val" in data_path:
            from .paths import val_path
            files={f.name:f for f in files}
            files=[files[name] for name in val_path]  # 
        elif "train" in data_path:
            from .paths import train_path
            files={f.name:f for f in files}
            files=[files[name] for name in train_path]  # 
        if self.sample_rate < 1:
            num_examples = round(int(len(files) * self.sample_rate))
            if num_examples>0:
                files = files[:num_examples]
            else:
                files = files[num_examples:]
        self.indices=[]
        start_point=0
        for file in sorted(files):
            metadata, num_slices = self._retrieve_metadata(file)
            self.examples += [
                (file, slice_ind) for slice_ind in range(num_slices)
            ]
            self.indices+=[[start_point+i for i in range(num_slices)]]
            start_point+=num_slices

        self.masks = [ProbMask(mask_config) for mask_config in mask_configs]
        self.files=files

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        file, slice_id = self.examples[item]
        with h5py.File(file, 'r') as data:
            full_kspace = np.array(data['kspace'][slice_id, ...])
        
        csm_path=os.path.join(self.data_path,f"../csm/{self.data_path.split('/')[-1]}",f"{file.name[:-3]}_{slice_id}.npz")
        if os.path.exists(csm_path):
            csm=np.load(csm_path,allow_pickle=True)['arr_0']
        else:
            csm=np.load(csm_path.replace(".npz",".npy"),allow_pickle=True)

        full_kspace = complex2real(full_kspace)
        csm = complex2real(csm)
        full_kspace = torch.from_numpy(full_kspace).float()
        csm = torch.from_numpy(csm).float()

        full_kspace = torch.view_as_complex(full_kspace)
        csm = torch.view_as_complex(csm).squeeze().permute(2,0,1)

        full_image=ifft2(full_kspace)
        full_image_cropped=T.center_crop(full_image,(320,320))
        full_kspace_cropped=fft2(full_image_cropped)
        
        csm=T.center_crop(csm,(320,320))

        return {0:full_kspace_cropped*1e6, 1:csm, 2:[mask.sample(item) for mask in self.masks], 3:file.name, 4:slice_id, 5:[mask.mask_prob for mask in self.masks]}
    
    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices
