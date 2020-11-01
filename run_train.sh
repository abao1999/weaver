python train.py \
  --data-test '/data/shared/abao/DNNTuples/train/*.h5' \
  --data-config 'data/ak8_points_pf_sv_hww_ptmasswgt_h5.yaml' \
  --network-config 'networks/particle_net_pf_sv.py' \
  --model-prefix 'models/testh5' \
  --train-val-split 0.8 \
  --start-lr 2e-2 \
  --num-epochs 20 \
  --optimizer ranger \
  --num-workers 4 \
  --batch-size 64 \
  --fetch-step 10 \
  --gpu '0,2,3,4,5' \
  --fetch-by-files \
  | tee 'logs/train.log'
