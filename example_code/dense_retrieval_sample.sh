
# DPR 학습 예시
python ./src/dense_retrieval.py \
--batch_size 4 \
--report_name BERT_neg3_bm2 \
--bm25 True \
--epochs 1 \
--num_neg 3 --bm_num 2 \
--dataset wiki \
--wandb False \
--test_query True \
--dpr_gradient_accumulation_steps 16
