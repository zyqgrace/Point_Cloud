
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

for model in 'rscnn'; do # 'rscnn' 'pointnet2' 'dgcnn' 'curvenet'; do
for cor in 'uniform' 'gaussian' 'background' 'impulse' 'upsampling'  'density'  'shear' 'rotation' 'cutout' 'distortion'; do
# 'distortion_rbf' 'distortion_rbf_inv' 'density_inc'
for sev in 1 2 3 4 5; do

# for aug in 'rsmix' 'cutmix_r' 'cutmix_k' 'mixup' 'pgd'; do

# CUDA_VISIBLE_DEVICES=0 python test.py --model-path runs/${aug}_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_${aug}_${cor}_${sev}.txt

# done

# for adapt in 'tent' 'bn'; do

# CUDA_VISIBLE_DEVICES=0 python  test.py  --model-path cor_exp/dgcnn_${model}_run_1/model_best_test.pth --exp-config configs/${adapt}/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_${adapt}_${cor}_${sev}.txt

# done

CUDA_VISIBLE_DEVICES=0 python test.py --model-path runs/rscnn_${model}_run_cutmix_r/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_cutmix_r_${cor}_${sev}.txt

done
done
done