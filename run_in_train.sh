python train.py \
  --data-train '/data/shared/abao/DNNTuples/train/*.h5' \
  --data-config 'data/ak8_in_pf_sv_hww_ptmasswgt_h5_h4q.yaml' \
  --network-config 'networks/in_pf_sv_2.py' \
  --model-prefix 'models/testh5' \
  --train-val-split 0.8 \
  --start-lr 2e-2 \
  --num-epochs 20 \
  --optimizer 'adam' \
  --num-workers 1 \
  --batch-size 64 \
  --fetch-step 10 \
  --gpu '0,2,3,4' \
  --fetch-by-files \
  | tee 'logs/train.log'
