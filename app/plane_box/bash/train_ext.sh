cd ~/ws

bash py.sh app/plane_box/train_corner_ext.py
sleep 60

bash py.sh app/plane_box/train_paralle_ext.py
sleep 60

bash py.sh app/plane_box/train_three_ext.py