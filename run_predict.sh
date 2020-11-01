python train.py \
  --predict \
  --data-test '/data/shared/abao/DNNTuples/test/*.h5' \
  --data-config 'data/ak8_points_pf_sv_hww_ptmasswgt_h5.yaml' \
  --network-config 'networks/particle_net_pf_sv.py' \
  --model-prefix 'models/testh5_epoch-19_state.pt' \
  --num-workers 1 \
  --batch-size 64 \
  --gpu '0,2' \
  --predict-output 'output/h5_model_test_ptwgt_h5_weight_epoch19_2gpus_1worker.root' \
  | tee 'logs/predict.log'
