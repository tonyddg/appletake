cd ~/ws

bash py.sh app/plane_box/train_corner_ply.py
sleep 60

bash py.sh app/plane_box/train_paralle_ply.py
sleep 60

bash py.sh app/plane_box/train_three_ply.py