.\.venv\Scripts\activate

#4-bit weights only
python train.py --epochs 30 --lr 0.01 --weight_decay 0.0001 --weight_bits 4 --activation_bits 32 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w4_a32

#6-bit both:
python train.py --epochs 30 --lr 0.01 --weight_decay 0.0001 --weight_bits 6 --activation_bits 6 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w6_a6

#4-bit both:
python train.py --epochs 30 --lr 0.01 --weight_decay 0.0001 --weight_bits 4 --activation_bits 4 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w4_a4

#2-bit both:
python train.py --epochs 30 --lr 0.001 --weight_decay 0.0001 --weight_bits 2 --activation_bits 2 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w2_a2

#4-bit activations 8-bit weights:
python train.py --epochs 30 --lr 0.1 --weight_decay 0.0001 --weight_bits 8 --activation_bits 4 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w8_a4

#8-bit activations 4-bit weights:
python train.py --epochs 30 --lr 0.01 --weight_decay 0.0001 --weight_bits 4 --activation_bits 8 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w4_a8

#8-bit both:
python train.py --epochs 30 --lr 0.001 --weight_decay 0.0001 --weight_bits 8 --activation_bits 8 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w8_a8

#4-bit activations 2-bit weights:
python train.py --epochs 30 --lr 0.1 --weight_decay 0.0001 --weight_bits 2 --activation_bits 4 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w2_a4

#2-bit activations 4-bit weights:
python train.py --epochs 30 --lr 0.01 --weight_decay 0.0001 --weight_bits 4 --activation_bits 2 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w4_a2

#6-bit activations 2-bit weights:
python train.py --epochs 30 --lr 0.001 --weight_decay 0.0001 --weight_bits 2 --activation_bits 6 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w2_a6

#2-bit activations 6-bit weights:
python train.py --epochs 30 --lr 0.05 --weight_decay 0.0001 --weight_bits 6 --activation_bits 2 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w6_a2

#3-bit both:
python train.py --epochs 30 --lr 0.005 --weight_decay 0.0001 --weight_bits 3 --activation_bits 3 --quantize_weights_during_forward --use_wandb --wandb_project cs6886_a3 --wandb_run_name w3_a3
