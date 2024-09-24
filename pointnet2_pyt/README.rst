Pointnet2/Pointnet++ PyTorch
============================

* Implemention of Pointnet2/Pointnet++ written in `PyTorch <http://pytorch.org>`_.

* Supports Multi-GPU via `nn.DataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel>`_.

* Supports PyTorch version >= 1.0.0.  Use `v1.0 <https://github.com/erikwijmans/Pointnet2_PyTorch/releases/tag/v1.0>`_
  for support of older versions of PyTorch.


See the official code release for the paper (in tensorflow), `charlesq34/pointnet2 <https://github.com/charlesq34/pointnet2>`_,
for official model definitions and hyper-parameters.

The custom ops used by Pointnet++ are currently **ONLY** supported on the GPU using CUDA.

Setup
-----

* Install ``python`` -- This repo is tested with ``2.7``, ``3.5``, and ``3.6``


* Install dependencies

  ::

    pip install -r requirements.txt


* Building `_ext` module

  ::

    python setup.py build_ext --inplace


* Optionally, you can also install this repo as a package

  ::

    pip install -e .


Example training
------------------

Two training examples are provided by ``pointnet2/train/train_sem_seg.py`` and ``pointnet2/train/train_cls.py``.
The datasets for both will be downloaded automatically by default.


They can be run via

::

  python -m pointnet2.train.train_cls

  python -m pointnet2.train.train_sem_seg


Both scripts will print training progress after every epoch to the command line.  Use the ``--visdom`` flag to
enable logging to visdom and more detailed logging of training progress.


Contributing
------------

This repository uses `black <https://github.com/ambv/black>`_ for linting and style enforcement on python code.
For c++/cuda code,
`clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_ is used for style.  The simplest way to
comply with style is via `pre-commit <https://pre-commit.com/>`_

::

  pip install pre-commit
  pre-commit install



###
 # @Description: 
 # @Autor: Jiachen Sun
 # @Date: 2022-02-16 22:23:16
 # @LastEditors: Jiachen Sun
 # @LastEditTime: 2022-02-23 17:20:27
### 
if [ ! -d "output" ]; then
    mkdir "output"
fi

for model in 'gdanet'; do #'pointnet' 'pct' 'rscnn' 'pointnet2'  'simpleview' 'dgcnn'  'pointMLP' 'curvenet'; do
for cor in 'uniform' 'gaussian' 'background' 'impulse' 'upsampling' 'distortion_rbf' 'distortion_rbf_inv' 'density' 'density_inc' 'shear' 'rotation' 'cutout' 'distortion'  'occlusion' 'lidar'; do

for sev in 1 2 3 4 5; do

# for aug in 'rsmix' 'cutmix_r' 'cutmix_k' 'mixup' 'pgd'; do

# CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/${aug}_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_${aug}_${cor}_${sev}.txt

# done

# for adapt in 'tent' 'bn'; do

# CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path cor_exp/dgcnn_${model}_run_1/model_best_test.pth --exp-config configs/${adapt}/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_${adapt}_${cor}_${sev}.txt

# done

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/dgcnn_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_none_${cor}_${sev}.txt

done
done
done


Citation
--------

::

  @article{pytorchpointnet++,
        Author = {Erik Wijmans},
        Title = {Pointnet++ Pytorch},
        Journal = {https://github.com/erikwijmans/Pointnet2_PyTorch},
        Year = {2018}
  }

  @inproceedings{qi2017pointnet++,
      title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
      author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
      booktitle={Advances in Neural Information Processing Systems},
      pages={5099--5108},
      year={2017}
  }
