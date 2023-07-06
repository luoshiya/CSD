python datafree_kd.py \
--method csd \
--dataset cifar10 \
--batch_size 128 \
--teacher wrn40_2 \
--student wrn16_1 \
--lr 0.1 \
--epochs 100 \
--kd_steps 2000 \
--ep_steps 2000 \
--g_steps 200 \
--synthesis_batch_size 200 \
--lr_g 1e-3 \
--csd 10 \
--adv 0 \
--bn 1.0 \
--oh 1.0 \
--gpu 0 \
--seed 1 \
--T 20 \
--save_dir run/csd/cifar10_wrn402_wrn161 \
--init_bank run/csd/cifar10_wrn402_initBank \
--run_kd