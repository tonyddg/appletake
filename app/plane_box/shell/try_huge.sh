cd ~/ws
source setup.sh

python app/plane_box/ply/train.py --env_type three app/plane_box/conf/tuned_huge_half_lr.yaml

sleep 60

python app/plane_box/ply/train.py --env_type three app/plane_box/conf/tuned_huge.yaml

sleep 60

python app/plane_box/ply/train.py --env_type corner app/plane_box/conf/tuned_huge.yaml
