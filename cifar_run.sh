# Normal Darknet53 cifar10 training:
python3 train_cifar_10.py \
    --exp cifar_10_msa_darknet53_paper_custom_epoch_350_2 \
    --epochs 350 \
    --lr 0.0005 \
    --batchsize 1024 \
    --model msa_darknet53

# Darknet53 MSA training:
python3 train_cifar_10.py \
    --exp cifar_10_darknet53_epoch_350_all_msas_attention_block_a_3 \
    --epochs 350 \
    --lr 0.0005 \
    --batchsize 1024 \
    --model msa_darknet53