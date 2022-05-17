# python3 graph.py \
#     --tb \
#     NEW/cifar_10_darknet53_epoch_350_all_msas_attention_block_a/events.out.tfevents.1648750271.pop-os.3703902.0 \
#     NEW/cifar_10_darknet53_paper_custom_epoch_350_3/events.out.tfevents.1648605097.pop-os.1833269.0 \
#     --labels \
#     "Darknet53 MSA" Darknet53 \
#     --title "Darknet53 vs Darknet53 MSA"\
#     --type classifier \
#     --out "darknet_vs_darknetmsa.png"

python3 graph.py \
    --tb \
    cifar_10_darknet53_epoch_350_all_msas_attention_block_a/events.out.tfevents.1648750271.pop-os.3703902.0 \
    cifar_10_darknet53_paper_custom_epoch_350_3/events.out.tfevents.1648605097.pop-os.1833269.0 \
    --labels \
    "Darknet53 MSA" Darknet53 \
    --title "Darknet53 vs Darknet53 MSA"\
    --type classifier \
    --out "darknet_vs_darknetmsa.png"