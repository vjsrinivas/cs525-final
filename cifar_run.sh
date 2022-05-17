#python3 train_cifar_100.py \
#    --exp cifar_100_darknet53_paper_custom_epoch_500 \
#    --epochs 500

#python3 train_cifar_10.py \
#    --exp cifar_10_darknet53_paper_custom_epoch_350 \
#    --epochs 350 \
#    --lr 0.00125 \
#    --batchsize 1024

#python3 train_cifar_10.py \
#    --exp cifar_10_msa_darknet53_paper_custom_epoch_350_2 \
#    --epochs 350 \
#    --lr 0.0005 \
#    --batchsize 1024 \
#    --model msa_darknet53

#python3 train_cifar_10.py \
#    --exp cifar_10_darknet53_paper_custom_epoch_350_3 \
#    --epochs 350 \
#    --lr 0.0005 \
#    --batchsize 1024

#python3 train_cifar_10.py \
#    --exp cifar_10_darknet53_paper_custom_epoch_350_2_msas \
#    --epochs 350 \
#    --lr 0.0005 \
#    --batchsize 1024 \
#    --model msa_darknet53

# python3 train_cifar_10.py \
#     --exp cifar_10_darknet53_epoch_350_all_msas_attention_block_a_2 \
#     --epochs 350 \
#     --lr 0.0005 \
#     --batchsize 1024 \
#     --model msa_darknet53

python3 train_cifar_10.py \
    --exp cifar_10_darknet53_epoch_350_all_msas_attention_block_a_3 \
    --epochs 350 \
    --lr 0.0005 \
    --batchsize 1024 \
    --model msa_darknet53