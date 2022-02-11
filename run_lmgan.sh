dataset_dir='dataset/ag/evenly_'
processor='ag'

label_nums=10
unlabel_nums=5000

batch_size=8
max_length=256

epochs=3

alpha=4

num_labels_in_batch=2
seed_val=100

python -u cleanse.py --dataset_dir='dataset/ag/' --label_num=$label_nums --unlabel_num=$unlabel_nums 
 
python -u main.py \
--label_num=$label_nums \
--dataset_dir=$dataset_dir \
--data_processor=$processor \
--batch_size=$batch_size \
--seed_val=$seed_val \
--max_length=$max_length \
--epochs=$epochs \
--alpha=$alpha \
--is_training='True' \
--b_num_label=$num_labels_in_batch\
