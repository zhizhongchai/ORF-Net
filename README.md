# ORF-Net
Implementation for the paper Deep Omni-supervised Learning for Rib Fracture Detection from Chest Radiology Images by Zhizhong Chai, [Luyang Luo](https://llyxc.github.io/), [Huangjing Lin](https://www.linkedin.com/in/huangjing-lin-3bb526a0/?originalSubdomain=hk), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/), and [Hao Chen](https://www.cse.ust.hk/admin/people/faculty/profile/jhc)

## Installation

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

*Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to Detectron2.*

Then build ORF-Net with:

```
git clone https://github.com/zhizhongchai/ORF-Net.git
cd ORF-Net
python setup.py build develop
```

### Train Your Own Models

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:

```
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --config-file configs/Omni/R_50_1x_omni_box_mask_dot_unlabel.yaml \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/box_mask_dot_unlabel/
```

### Acknowledgement
The codes are modified from [AdelaiDet]((https://github.com/aim-uofa/AdelaiDet)).
