if [ "$1" = "eth" ];then
  echo 'dataset: eth'
  python run.py \
  --dataset "$1" \
  --data_dir "./dataset/" \
  --epoch 350 \
  --lr 0.001 \
  --lr_scheduler 1 \
  --lr_milestones 55 \
  --lr_gamma 0.5 \
  --obs_len 8 \
  --pred_len 12 \
  --train_batch_size 512 \
  --test_batch_size 4096 \
  --seed 1 \
  --cuda \
  --gpu_num 0 \
  --checkpoint "./checkpoint/" \
  --end_centered \
  --data_scaling 1.9 0.4 \
  --split_thea 4 \
  --split_temporal_interval 4 \
  --tree_degree 3 \
  --num_k 20
elif [ "$1" = "hotel" ];then
  echo 'dataset: hotel'
  python run.py \
  --dataset "$1" \
  --data_dir "./dataset/" \
  --epoch 350 \
  --lr 0.001 \
  --lr_scheduler -1 \
  --lr_gamma 0.5 \
  --obs_len 8 \
  --pred_len 12 \
  --train_batch_size 512 \
  --test_batch_size 4096 \
  --seed 1 \
  --cuda \
  --gpu_num 0 \
  --checkpoint "./checkpoint/" \
  --end_centered \
  --data_flip \
  --data_scaling 1.0 1.0 \
  --split_thea 6 \
  --split_temporal_interval 4 \
  --tree_degree 3 \
  --num_k 20
elif [ "$1" = "univ" ];then
  echo 'dataset: univ'
  python run.py \
  --dataset "$1" \
  --data_dir "./dataset/" \
  --epoch 350 \
  --lr 0.003 \
  --lr_scheduler 0 \
  --lr_milestones 150 \
  --lr_gamma 0.5 \
  --obs_len 8 \
  --pred_len 12 \
  --train_batch_size 256 \
  --test_batch_size 4096 \
  --seed 1 \
  --cuda \
  --gpu_num 1 \
  --checkpoint "./checkpoint/" \
  --end_centered \
  --data_scaling 1.0 1.0 \
  --split_thea 4 \
  --split_temporal_interval 4 \
  --tree_degree 3 \
  --num_k 20
elif [ "$1" = "zara1" ];then
  echo 'dataset: zara1'
  python run.py \
  --dataset "$1" \
  --data_dir "./dataset/" \
  --epoch 350 \
  --lr 0.001 \
  --lr_scheduler 0 \
  --lr_milestones 200 300 \
  --lr_gamma 0.5 \
  --obs_len 8 \
  --pred_len 12 \
  --train_batch_size 512 \
  --test_batch_size 4096 \
  --seed 1 \
  --cuda \
  --gpu_num 2 \
  --checkpoint "./checkpoint/" \
  --end_centered \
  --data_scaling 1.0 1.0 \
  --split_thea 12 \
  --split_temporal_interval 4 \
  --tree_degree 3 \
  --num_k 20
elif [ "$1" = "zara2" ];then
  echo 'dataset: zara2'
  python run.py \
  --dataset "$1" \
  --data_dir "./dataset/" \
  --epoch 350 \
  --lr 0.001 \
  --lr_scheduler 0 \
  --lr_milestones 50 150 250 \
  --lr_gamma 0.5 \
  --obs_len 8 \
  --pred_len 12 \
  --train_batch_size 512 \
  --test_batch_size 4096 \
  --seed 1 \
  --cuda \
  --gpu_num 3 \
  --checkpoint "./checkpoint/" \
  --end_centered \
  --data_scaling 1.0 1.0 \
  --split_thea 4 \
  --split_temporal_interval 4 \
  --tree_degree 3 \
  --num_k 20
elif [ "$1" = "sdd" ];then
  echo 'dataset: sdd'
  python run.py \
  --dataset "$1" \
  --data_dir "./dataset/" \
  --epoch 500 \
  --lr 0.003 \
  --lr_scheduler 0 \
  --lr_milestones 270 400 \
  --lr_gamma 0.5 \
  --obs_len 8 \
  --pred_len 12 \
  --train_batch_size 512 \
  --test_batch_size 4096 \
  --seed 1 \
  --cuda \
  --gpu_num 4 \
  --checkpoint "./checkpoint/" \
  --end_centered \
  --data_scaling 1.0 1.0 \
  --split_thea 4 \
  --split_temporal_interval 4 \
  --tree_degree 3 \
  --num_k 20
else
  echo "Please input dataset name like this: sh train.sh 'eth'"
fi


