cd ~/ws

bash py.sh app/plane_box/finetuning/corner_hard_vec_sac_raw.py
sleep 60

bash py.sh app/plane_box/finetuning/paralle_hard_vec_sac_raw.py
sleep 60

bash py.sh app/plane_box/finetuning/three_hard_vec_sac_raw.py