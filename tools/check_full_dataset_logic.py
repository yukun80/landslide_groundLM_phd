"""用真实HDF5小样例验证正样本清洗、固定身份及空间边界。"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image

from xiaomi_lens.full_dataset import prepare as p
from xiaomi_lens.full_dataset.common import NAMES,atomic_write_json,ROOT
from xiaomi_lens.full_dataset.guards import unsupported_color


def main():
    results=[]
    with tempfile.TemporaryDirectory(prefix='full-dataset-check-') as tmp:
        tmp=Path(tmp)
        for case in ('tiny_positive','background','invalid','illegal','constant'):
            folder=tmp/case;folder.mkdir();mask=np.zeros((9,11),dtype='uint8');valid=np.ones_like(mask)
            image=np.arange(3*9*11,dtype='float32').reshape(3,9,11)
            if case!='background':mask[0,0]=1
            if case=='invalid':valid[:]=0
            if case=='illegal':mask[2,2]=2
            if case=='constant':image[:]=0
            paths={'mask_hdf5':folder/'mask.h5','image_hdf5':folder/'image.h5'}
            with h5py.File(paths['mask_hdf5'],'w') as f:f['mask']=mask;f['valid_mask']=valid
            with h5py.File(paths['image_hdf5'],'w') as f:f['image']=image
            row={'sample_key':case,'source_origin':'original',**{k:str(v) for k,v in paths.items()}}
            with patch.object(p,'source_paths',return_value=paths):
                first=p.inspect_one((NAMES[0],row),folder)
                second=p.inspect_one((NAMES[0],row),folder)
            assert first==second
            assert first['status']==('eligible' if case=='tiny_positive' else 'excluded'),(case,first)
            if case=='tiny_positive':assert first['positive_pixels']==1
            results.append({'case':case,'status':first['status'],'reason':first.get('reason')})
        index=p.SpatialIndex();index.add({'id':'a','footprint':[0,0,1,1]},'train')
        assert not index.query([1,0,2,1]) # 共边不是重叠
        assert index.query([.99,0,2,1])
        assert p.bounds({'geotransform':[2,0,10,0,-3,40]},(5,7))==[10,25,24,40]
        assert p.group_key(NAMES[0],{'sample_key':'train_1','source_origin':'original','scene':'1'})!=p.group_key(NAMES[0],{'sample_key':'test_1','source_origin':'original','scene':'1'})
        rgb=tmp/'rgb.png';mask=tmp/'mask.png'
        Image.fromarray(np.full((8,8,3),230,dtype='uint8')).save(rgb)
        Image.fromarray(np.full((8,8),255,dtype='uint8')).save(mask)
        assert unsupported_color('目标整体为灰绿色。',rgb,mask)
        assert not unsupported_color('目标呈白色，与周围绿色形成对比。',rgb,mask)
    atomic_write_json(ROOT/'outputs/full-dataset-logic-check.json',{'passed':True,'actual_hdf5_cases':results,'spatial_boundaries_checked':True})
    print('PASS',results,flush=True)


if __name__=='__main__':main()
