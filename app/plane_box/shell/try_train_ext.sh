cd ~/ws
source setup.sh

python app/plane_box/ext/train_ext.py --net_type eff --env_type three --lr_schedule restart --epoch_scale 1.2 --lr 1e-3 --batch_size 64 --num_worker 4 --weight_decay 3e-6
sleep 60

python app/plane_box/ext/train_ext.py --net_type eff --env_type paralle --lr_schedule restart --epoch_scale 1.2 --lr 1e-3 --batch_size 64 --num_worker 4 --weight_decay 3e-6
sleep 60

python app/plane_box/ext/train_ext.py --net_type eff --env_type corner --lr_schedule restart --epoch_scale 1.2 --lr 1e-3 --batch_size 64 --num_worker 4 --weight_decay 3e-6
sleep 60
