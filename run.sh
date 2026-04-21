# 统计运行时间
python timing_analyzer.py --model roberta-large --dataset 20_newsgroups --max_length 256 --batch_size 16 --lr 1e-3 


nohup python -u main_.py --label --partition --model roberta-large --method raw --GPU 0 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 50 --lr 1e-2 > result/roberta_20news_iid_raw.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method HeLoRA --GPU 0 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 50 --lr 1e-2 > result/roberta_20news_iid_helora.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method pq --GPU 0 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 50 --lr 1e-2 > result/roberta_20news_iid_pq.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method topk --GPU 0 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 50 --lr 1e-2 > result/roberta_20news_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method FFTHM --GPU 0 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 50 --lr 1e-2 > result/roberta_20news_iid_ffthm.out 2>&1 &

nohup python -u main_.py --label --partition --model distilbert-base-multilingual-cased --method motivation --GPU 0 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_motivation_w.out 2>&1 &
nohup python -u main_.py --label --partition --model distilbert-base-multilingual-cased --method motivation --GPU 1 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 --pmax 0.7 --pmin 0.6 > result/distilbert_20news_iid_motivation_s.out 2>&1 &
nohup python -u main_.py --label --partition --method motivation --GPU 0 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/roberta_20news_iid_motivation_w.out 2>&1 &
nohup python -u main_.py --label --partition --method motivation --GPU 1 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 --pmax 0.7 --pmin 0.6 > result/roberta_20news_iid_motivation_s.out 2>&1 &


nohup python -u main_.py --label --partition --model distilbert-base-multilingual-cased --method raw --GPU 0 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_raw.out 2>&1 &
nohup python -u main_.py --label --partition --model distilbert-base-multilingual-cased --method new1 --GPU 0 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_new.out 2>&1 &
# 行和列均匀冻结
nohup python -u main_.py --label --partition --update_proportion 0.5 --model distilbert-base-multilingual-cased --method new1 --GPU 1 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_new5.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.6 --model distilbert-base-multilingual-cased --method new1 --GPU 2 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_new6.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.7 --model distilbert-base-multilingual-cased --method new1 --GPU 3 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_new7.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.8 --model distilbert-base-multilingual-cased --method new1 --GPU 4 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_new8.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.9 --model distilbert-base-multilingual-cased --method new1 --GPU 5 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_new9.out 2>&1 &

nohup python -u main_.py --label --partition --update_proportion 0.5 --model distilbert-base-multilingual-cased --method new1 --GPU 1 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_test5.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.6 --model distilbert-base-multilingual-cased --method new1 --GPU 2 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_test6.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.7 --model distilbert-base-multilingual-cased --method new1 --GPU 3 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_test7.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.8 --model distilbert-base-multilingual-cased --method new1 --GPU 4 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_test8.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.9 --model distilbert-base-multilingual-cased --method new1 --GPU 5 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_test9.out 2>&1 &

# 只冻结行
nohup python -u main_.py --label --partition --update_proportion 0.5 --model distilbert-base-multilingual-cased --method new1 --GPU 1 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_nnew5.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.6 --model distilbert-base-multilingual-cased --method new1 --GPU 2 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_nnew6.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.7 --model distilbert-base-multilingual-cased --method new1 --GPU 3 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_nnew7.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.8 --model distilbert-base-multilingual-cased --method new1 --GPU 4 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_nnew8.out 2>&1 &
nohup python -u main_.py --label --partition --update_proportion 0.9 --model distilbert-base-multilingual-cased --method new1 --GPU 5 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_nnew9.out 2>&1 &

nohup python -u main_.py --label --partition --model distilbert-base-multilingual-cased --method block_opt --GPU 1 --lr 1e-3 --max_length 256 --batch_size 16 --comm_round 100 > result/distilbert_20news_iid_block.out 2>&1 &


nohup python -u main_.py --label --partition --model llama-3.2-1B --method compeft --GPU 0 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/3llama_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method compeft --GPU 1 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/3llama_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-2-7B --method compeft --GPU 2 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/llama_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --model llama-2-7B --method compeft --GPU 3 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/llama_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --GPU 4 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 > result/roberta_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --method compeft --GPU 4 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 > result/roberta_20news_niid_compeft.out 2>&1 &


nohup python -u main_.py --label --partition --model llama-3.2-1B --method updateW --optimize 1 --GPU 0 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 --point 110 > result/3llama_20news_iid_optim.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method updateW --GPU 1 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 --point 51 > result/3llama_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method block_opt --GPU 4 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/3llama_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method compeft --GPU 0 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/3llama_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method topk_AB --GPU 3 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/3llama_20news_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method prune --GPU 2 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/3llama_20news_iid_prune.out 2>&1 &

nohup python -u main_.py --label --model llama-3.2-1B --method updateW --optimize 1 --GPU 0 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 --point 110 > result/3llama_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method updateW --GPU 1 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 --point 51 > result/3llama_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method block_opt --GPU 4 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/3llama_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method compeft --GPU 1 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/3llama_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method topk_AB --GPU 3 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/3llama_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method prune --GPU 6 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/3llama_20news_niid_prune.out 2>&1 &

nohup python -u main_.py --label --partition --model llama-2-7B --method updateW --optimize 1 --GPU 0 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 --point 110 > result/llama_20news_iid_optim.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-2-7B --method updateW --GPU 1 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 --point 51 > result/llama_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-2-7B --method block_opt --GPU 4 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/llama_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-2-7B --method compeft --GPU 2 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/llama_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-2-7B --method topk_AB --GPU 3 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/llama_20news_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-2-7B --method prune --GPU 2 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/llama_20news_iid_prune.out 2>&1 &

nohup python -u main_.py --label --model llama-2-7B --method updateW --optimize 1 --GPU 0 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 --point 110 > result/llama_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --model llama-2-7B --method updateW --GPU 6 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 --point 51 > result/llama_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --model llama-2-7B --method block_opt --GPU 7 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/llama_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --model llama-2-7B --method compeft --GPU 3 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/llama_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --model llama-2-7B --method topk_AB --GPU 3 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/llama_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --model llama-2-7B --method prune --GPU 2 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 20 > result/llama_20news_niid_prune.out 2>&1 &

nohup python -u main_.py --label --partition --method updateW --optimize 1 --GPU 1 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 --point 110 > result/roberta_20news_iid_optim.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --GPU 2 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 --point 110 > result/roberta_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method block_opt --GPU 3 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 > result/roberta_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --GPU 4 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 > result/roberta_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method topk_AB --GPU 4 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 > result/roberta_20news_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --GPU 7 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 > result/roberta_20news_iid_prune.out 2>&1 &

nohup python -u main_.py --label --method updateW --optimize 1 --GPU 1 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 --point 110 > result/roberta_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --method updateW --GPU 2 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 --point 110 > result/roberta_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method block_opt --GPU 3 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 > result/roberta_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --method compeft --GPU 4 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 > result/roberta_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --method topk_AB --GPU 1 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 > result/roberta_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --method prune --GPU 7 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.02 --weight 0.01 --comm_round 50 > result/roberta_20news_niid_prune.out 2>&1 &



nohup python -u main_.py --label --model llama-3.2-1B --method updateW --dataset glue --subdataset cola --optimize 1 --GPU 0 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 --point 110 > result/cola_optim.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method block_opt --dataset glue --subdataset cola --GPU 1 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/cola_block.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method compeft --dataset glue --subdataset cola --GPU 4 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/cola_compeft.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method topk_AB --dataset glue --subdataset cola --GPU 5 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/cola_topk.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method prune --dataset glue --subdataset cola --GPU 7 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/cola_prune.out 2>&1 &

nohup python -u main_.py --label --model llama-3.2-1B --method updateW --dataset glue --subdataset sst2 --optimize 1 --GPU 0 --lr 1e-4 --max_length 128 --batch_size 16 --comm_round 10 --point 110 > result/sst2_optim.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method block_opt --dataset glue --subdataset sst2 --GPU 1 --lr 1e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/sst2_block.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method compeft --dataset glue --subdataset sst2 --GPU 2 --lr 1e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/sst2_compeft.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method topk_AB --dataset glue --subdataset sst2 --GPU 3 --lr 1e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/sst2_topk.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method prune --dataset glue --subdataset sst2 --GPU 4 --lr 1e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/sst2_prune.out 2>&1 &

nohup python -u main_.py --label --model llama-3.2-1B --method updateW --dataset glue --subdataset qqp --optimize 1 --GPU 0 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 --point 110 > result/qqp_optim.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method block_opt --dataset glue --subdataset qqp --GPU 1 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/qqp_block.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method compeft --dataset glue --subdataset qqp --GPU 4 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/qqp_compeft.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method topk_AB --dataset glue --subdataset qqp --GPU 5 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/qqp_topk.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method prune --dataset glue --subdataset qqp --GPU 6 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/qqp_prune.out 2>&1 &

nohup python -u main_.py --label --model llama-3.2-1B --method updateW --dataset glue --subdataset mrpc --optimize 1 --GPU 2 --lr 5e-4 --max_length 128 --batch_size 16 --comm_round 10 --point 110 > result/mrpc_optim.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method block_opt --dataset glue --subdataset mrpc --GPU 3 --lr 5e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/mrpc_block.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method compeft --dataset glue --subdataset mrpc --GPU 7 --lr 5e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/mrpc_compeft.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method topk_AB --dataset glue --subdataset mrpc --GPU 5 --lr 5e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/mrpc_topk.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method prune --dataset glue --subdataset mrpc --GPU 6 --lr 5e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/mrpc_prune.out 2>&1 &

nohup python -u main_.py --model llama-3.2-1B --method updateW --dataset glue --subdataset stsb --optimize 1 --GPU 2 --lr 1e-3 --factor 2 --max_length 128 --weight 0.02 --batch_size 16 --comm_round 10 --point 110 > result/stsb_optim.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method block_opt --dataset glue --subdataset stsb --GPU 4 --lr 1e-3 --max_length 128 --weight 0 --batch_size 16 --comm_round 10 > result/stsb_block.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method compeft --dataset glue --subdataset stsb --GPU 7 --lr 1e-3 --max_length 128 --weight 0 --batch_size 16 --comm_round 10 > result/stsb_compeft.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method topk_AB --dataset glue --subdataset stsb --GPU 0 --lr 1e-3 --max_length 128 --weight 0 --batch_size 16 --comm_round 10 > result/stsb_topk.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method prune --dataset glue --subdataset stsb --GPU 1 --lr 1e-3 --max_length 128 --weight 0 --batch_size 16 --comm_round 10 > result/stsb_prune.out 2>&1 &

nohup python -u main_.py --label --model llama-3.2-1B --method updateW --dataset glue --subdataset mnli --optimize 1 --GPU 2 --lr 1e-4 --max_length 128 --batch_size 16 --comm_round 10 --point 110 > result/mnli_optim.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method block_opt --dataset glue --subdataset mnli --GPU 3 --lr 1e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/mnli_block.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method compeft --dataset glue --subdataset mnli --GPU 7 --lr 1e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/mnli_compeft.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method topk_AB --dataset glue --subdataset mnli --GPU 0 --lr 1e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/mnli_topk.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method prune --dataset glue --subdataset mnli --GPU 1 --lr 1e-4 --max_length 128 --batch_size 16 --comm_round 10 > result/mnli_prune.out 2>&1 &

nohup python -u main_.py --label --model llama-3.2-1B --method updateW --dataset glue --subdataset qnli --optimize 1 --GPU 4 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 --point 110 > result/qnli_optim.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method block_opt --dataset glue --subdataset qnli --GPU 5 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/qnli_block.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method compeft --dataset glue --subdataset qnli --GPU 6 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/qnli_compeft.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method topk_AB --dataset glue --subdataset qnli --GPU 0 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/qnli_topk.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method prune --dataset glue --subdataset qnli --GPU 1 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/qnli_prune.out 2>&1 &

nohup python -u main_.py --model llama-3.2-1B --method updateW --dataset squad --optimize 1 --GPU 2 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 --point 110 > result/squad_optim.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method block_opt --dataset squad --GPU 3 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/squad_block.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method compeft --dataset squad --GPU 4 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/squad_compeft.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method topk_AB --dataset squad --GPU 6 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/squad_topk.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method prune --dataset squad --GPU 7 --lr 1e-3 --max_length 128 --batch_size 16 --comm_round 10 > result/squad_prune.out 2>&1 &


nohup python -u main_.py --label --partition --method updateW --optimize 1 --GPU 1 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.05 --factor 1.5 --weight 0 --comm_round 50 --point 2 > result/distilbert_20news_iid_optim.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --GPU 2 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.05 --factor 1.5 --weight 0 --comm_round 50 --point 2 > result/distilbert_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method block_opt --GPU 3 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.05 --weight 0.05 --comm_round 50 > result/distilbert_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --GPU 4 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.05 --weight 0.05 --comm_round 50 > result/distilbert_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method topk_AB --GPU 5 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.05 --weight 0.05 --comm_round 50 > result/distilbert_20news_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --GPU 7 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.05 --weight 0.05 --comm_round 50 > result/distilbert_20news_iid_prune.out 2>&1 &

nohup python -u main_.py --label --method updateW --optimize 1 --GPU 1 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.05 --factor 1.5 --weight 0 --comm_round 50 --point 2 > result/distilbert_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --method updateW --GPU 2 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.05 --factor 1.5 --weight 0 --comm_round 50 --point 2 > result/distilbert_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method block_opt --GPU 3 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.05 --weight 0.05 --comm_round 50 > result/distilbert_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --method compeft --GPU 4 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.05 --weight 0.05 --comm_round 50 > result/distilbert_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --method topk_AB --GPU 5 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.05 --weight 0.05 --comm_round 50 > result/distilbert_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --method prune --GPU 7 --lr 1e-3 --max_length 256 --batch_size 16 --proportion 0.05 --weight 0.05 --comm_round 50 > result/distilbert_20news_niid_prune.out 2>&1 &


nohup python -u main_.py --label --partition --model roberta-large --method updateW --optimize 1 --GPU 1 --lr 1e-3 --weight 0.01 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 --point 110 > result/largeroberta_20news_iid_optim.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method updateW --GPU 2 --lr 1e-3 --weight 0.01 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 --point 110 > result/largeroberta_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method block_opt --GPU 3 --lr 1e-3 --weight 0.01 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 > result/largeroberta_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method compeft --GPU 4 --lr 1e-3 --weight 0.01 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 > result/largeroberta_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method topk_AB --GPU 5 --lr 1e-3 --weight 0.01 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 > result/largeroberta_20news_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method prune --GPU 7 --lr 1e-3 --weight 0.01 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 > result/largeroberta_20news_iid_prune.out 2>&1 &




nohup python -u main_.py --label --method updateW --optimize 1 --GPU 1 --lr 1e-3 --weight 0.02 --proportion 0.02 --max_length 350 --batch_size 16 --comm_round 50 --point 110 > result/distilbert_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --method updateW --GPU 2 --lr 1e-3 --weight 0.02 --proportion 0.02 --max_length 350 --batch_size 16 --comm_round 50 --point 110 > result/distilbert_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method block_opt --GPU 3 --lr 1e-3 --weight 0.02 --proportion 0.02 --max_length 350 --batch_size 16 --comm_round 50 > result/distilbert_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --method compeft --GPU 4 --lr 1e-3 --weight 0.02 --proportion 0.02 --max_length 350 --batch_size 16 --comm_round 50 > result/distilbert_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --method topk_AB --GPU 5 --lr 1e-3 --weight 0.02 --proportion 0.02 --max_length 350 --batch_size 16 --comm_round 50 > result/distilbert_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --method prune --GPU 7 --lr 1e-3 --weight 0.02 --proportion 0.02 --max_length 350 --batch_size 16 --comm_round 50 > result/distilbert_20news_niid_prune.out 2>&1 &



nohup python -u main_.py --label --model roberta-large --method updateW --optimize 1 --GPU 1 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 --point 110 > result/largeroberta_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method updateW --GPU 2 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 --point 110 > result/largeroberta_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method block_opt --GPU 3 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 > result/largeroberta_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method compeft --GPU 4 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 > result/largeroberta_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method topk_AB --GPU 5 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 > result/largeroberta_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method prune --GPU 6 --lr 1e-3 --max_length 128 --batch_size 16 --proportion 0.02 --comm_round 30 > result/largeroberta_20news_niid_prune.out 2>&1 &






nohup python -u main_.py --label --model roberta-large --method updateW --optimize 1 --GPU 0 --lr 1e-3 --proportion 0.02 --max_length 512 --batch_size 16 --comm_round 30 --point 110 > result/largeroberta_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method updateW --GPU 3 --lr 1e-3 --proportion 0.02 --max_length 512 --batch_size 16 --comm_round 30 --point 110 > result/largeroberta_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method block_opt --GPU 4 --lr 1e-3 --proportion 0.02 --max_length 512 --batch_size 16 --comm_round 30 > result/largeroberta_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method compeft --GPU 5 --lr 1e-3 --proportion 0.02 --max_length 512 --batch_size 16 --comm_round 30 > result/largeroberta_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method topk_AB --GPU 1 --lr 1e-3 --proportion 0.02 --max_length 512 --batch_size 16 --comm_round 30 > result/largeroberta_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method prune --GPU 2 --lr 1e-3 --proportion 0.02 --max_length 512 --batch_size 16 --comm_round 30 > result/largeroberta_20news_niid_prune.out 2>&1 &


nohup python -u main_.py --label --method updateW --optimize 1 --GPU 7 --lr 1e-3 --proportion 0.05 --max_length 512 --batch_size 16 --comm_round 50 --point 110 > result/distilbert_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --method updateW --GPU 7 --lr 1e-3 --proportion 0.05 --max_length 512 --batch_size 16 --comm_round 50 --point 110 > result/distilbert_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method block_opt --GPU 7 --lr 1e-3 --proportion 0.05 --max_length 512 --batch_size 16 --comm_round 50 > result/distilbert_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --method compeft --GPU 7 --lr 1e-3 --proportion 0.05 --max_length 512 --batch_size 16 --comm_round 50 > result/distilbert_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --method topk_AB --GPU 7 --lr 1e-3 --proportion 0.05 --max_length 512 --batch_size 16 --comm_round 50 > result/distilbert_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --method prune --GPU 7 --lr 1e-3 --proportion 0.05 --max_length 512 --batch_size 16 --comm_round 50 > result/distilbert_20news_niid_prune.out 2>&1 &



nohup python -u main_.py --label --partition --model roberta-large --method updateW --optimize 1 --GPU 0 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 30 --point 110 > result/largeroberta_20news_iid_optim.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method updateW --GPU 3 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 30 --point 110 > result/largeroberta_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method block_opt --GPU 4 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 30 > result/largeroberta_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method compeft --GPU 5 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 30 > result/largeroberta_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method topk_AB --GPU 1 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 30 > result/largeroberta_20news_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method prune --GPU 2 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 40 > result/largeroberta_20news_iid_prune.out 2>&1 &

nohup python -u main_.py --label --partition --method updateW --optimize 1 --GPU 6 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --point 110 > result/distilbert_20news_iid_optim.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --GPU 6 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --point 110 > result/distilbert_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method block_opt --GPU 6 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/distilbert_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --GPU 7 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/distilbert_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method topk_AB --GPU 7 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/distilbert_20news_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --GPU 7 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 60 > result/distilbert_20news_iid_prune.out 2>&1 &

nohup python -u main_.py --label --method updateW --optimize 1 --GPU 6 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --point 110 > result/distilbert_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --method updateW --GPU 6 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --point 110 > result/distilbert_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method block_opt --GPU 6 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/distilbert_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --method compeft --GPU 7 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/distilbert_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --method topk_AB --GPU 7 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/distilbert_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --method prune --GPU 7 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 60 > result/distilbert_20news_niid_prune.out 2>&1 &




nohup python -u main_.py --label --method topk_AB --dataset glue --subdataset cola --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 0 > result/cola_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --dataset glue --subdataset cola --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 1 --factor 1 --point 110 > result/cola_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --dataset glue --subdataset cola --model llama-3.2-1B --comm_round 16 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 2 > result/cola_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --dataset glue --subdataset cola --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 3 > result/cola_compeft.out 2>&1 &
nohup python -u main_.py --label --method block_opt --dataset glue --subdataset cola --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 4 > result/cola_block.out 2>&1 &

nohup python -u main_.py --label --method topk_AB --dataset glue --subdataset sst2 --model llama-3.2-1B --comm_round 10 --lr 1e-4 --max_length 128 --batch_size 32 --GPU 0 > result/sst2_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --dataset glue --subdataset sst2 --model llama-3.2-1B --comm_round 10 --lr 1e-4 --max_length 128 --batch_size 32 --GPU 1 --factor 1 --point 110 > result/sst2_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --dataset glue --subdataset sst2 --model llama-3.2-1B --comm_round 16 --lr 1e-4 --max_length 128 --batch_size 32 --GPU 2 > result/sst2_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --dataset glue --subdataset sst2 --model llama-3.2-1B --comm_round 10 --lr 1e-4 --max_length 128 --batch_size 32 --GPU 3 > result/sst2_compeft.out 2>&1 &
nohup python -u main_.py --label --method block_opt --dataset glue --subdataset sst2 --model llama-3.2-1B --comm_round 10 --lr 1e-4 --max_length 128 --batch_size 32 --GPU 4 > result/sst2_block.out 2>&1 &

nohup python -u main_.py --label --method topk_AB --dataset glue --subdataset mrpc --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 0 > result/mrpc_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --dataset glue --subdataset mrpc --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 1 --factor 1 --point 110 > result/mrpc_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --dataset glue --subdataset mrpc --model llama-3.2-1B --comm_round 16 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 2 > result/mrpc_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --dataset glue --subdataset mrpc --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 3 > result/mrpc_compeft.out 2>&1 &
nohup python -u main_.py --label --method block_opt --dataset glue --subdataset mrpc --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 4 > result/mrpc_block.out 2>&1 &

nohup python -u main_.py --method topk_AB --dataset glue --subdataset stsb --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 0 > result/stsb_topk_AB.out 2>&1 &
nohup python -u main_.py --method updateW --dataset glue --subdataset stsb --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 1 --factor 1 --point 110 > result/stsb_updateW.out 2>&1 &
nohup python -u main_.py --method prune --dataset glue --subdataset stsb --model llama-3.2-1B --comm_round 16 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 2 > result/stsb_prune.out 2>&1 &
nohup python -u main_.py --method compeft --dataset glue --subdataset stsb --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 3 > result/stsb_compeft.out 2>&1 &
nohup python -u main_.py --method block_opt --dataset glue --subdataset stsb --model llama-3.2-1B --comm_round 10 --lr 1e-3 --max_length 512 --batch_size 16 --GPU 4 > result/stsb_block.out 2>&1 &

nohup python -u main_.py --label --method topk_AB --dataset glue --subdataset qqp --model llama-3.2-1B --comm_round 10 --lr 1e-4 --max_length 128 --batch_size 32 --GPU 0 > result/qqp_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --dataset glue --subdataset qqp --model llama-3.2-1B --comm_round 10 --lr 1e-4 --max_length 128 --batch_size 32 --GPU 1 --factor 1 --point 110 > result/qqp_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --dataset glue --subdataset qqp --model llama-3.2-1B --comm_round 16 --lr 1e-4 --max_length 128 --batch_size 32 --GPU 2 > result/qqp_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --dataset glue --subdataset qqp --model llama-3.2-1B --comm_round 10 --lr 1e-4 --max_length 128 --batch_size 32 --GPU 3 > result/qqp_compeft.out 2>&1 &
nohup python -u main_.py --label --method block_opt --dataset glue --subdataset qqp --model llama-3.2-1B --comm_round 10 --lr 1e-4 --max_length 128 --batch_size 32 --GPU 4 > result/qqp_block.out 2>&1 &




nohup python -u main_.py --label --model llama-3.2-1B --method updateW --optimize 1 --GPU 0 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 20 --factor 1 --point 110 > result/llama_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method updateW --GPU 3 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 20 --factor 1 --point 51 > result/llama_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method block_opt --GPU 4 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 20 > result/llama_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method compeft --GPU 5 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 20 > result/llama_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method topk_AB --GPU 1 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 20 > result/llama_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method prune --GPU 2 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 30 > result/llama_20news_niid_prune.out 2>&1 &


nohup python -u main_.py --label --model roberta-large --method updateW --optimize 1 --GPU 7 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --factor 1 --point 110 > result/largeroberta_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method updateW --GPU 2 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --factor 1 --point 110 > result/largeroberta_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method block_opt --GPU 3 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/largeroberta_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method compeft --GPU 4 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/largeroberta_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method topk_AB --GPU 5 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/largeroberta_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method prune --GPU 6 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 60 > result/largeroberta_20news_niid_prune.out 2>&1 &






nohup python -u main_.py --label --partition --method topk_AB --model llama-3.2-1B --max_length 200 --GPU 0 > result/llama_20news_iid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --model llama-3.2-1B --max_length 200 --GPU 1 > result/llama_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method block_opt --model llama-3.2-1B --max_length 200 --GPU 2 > result/llama_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --model llama-3.2-1B --max_length 200 --GPU 3 > result/llama_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --model llama-3.2-1B --max_length 200 --GPU 4 > result/llama_20news_iid_prune.out 2>&1 &

nohup python -u main_.py --label --method topk_AB --model llama-3.2-1B --max_length 200 --GPU 5 > result/llama_20news_niid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --model llama-3.2-1B --max_length 200 --GPU 6 > result/llama_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method block_opt --model llama-3.2-1B --max_length 200 --GPU 7 > result/llama_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --method compeft --model llama-3.2-1B --max_length 200 --GPU 3 > result/llama_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --method prune --model llama-3.2-1B --max_length 200 --GPU 4 > result/llama_20news_niid_prune.out 2>&1 &

nohup python -u main_.py --label --partition --method updateW --model roberta-large --GPU 2 > result/largeroberta_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method block_opt --model roberta-large --GPU 0  > result/largeroberta_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --model roberta-large --GPU 4  > result/largeroberta_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method topk_AB --model roberta-large --GPU 5  > result/largeroberta_20news_iid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --model roberta-large --GPU 6  > result/largeroberta_20news_iid_prune.out 2>&1 &

nohup python -u main_.py --label --method updateW --model roberta-large --GPU 0 > result/largeroberta_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method block_opt --model roberta-large --GPU 2  > result/largeroberta_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --method compeft --model roberta-large --GPU 4  > result/largeroberta_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --method topk_AB --model roberta-large --GPU 5  > result/largeroberta_20news_niid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method prune --model roberta-large --GPU 6  > result/largeroberta_20news_niid_prune.out 2>&1 &

nohup python -u main_.py --label --partition --method updateW --GPU 7 > result/distilbert_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method block_opt --GPU 7  > result/distilbert_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --GPU 7  > result/distilbert_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method topk_AB --GPU 7  > result/distilbert_20news_iid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --GPU 7  > result/distilbert_20news_iid_prune.out 2>&1 &

nohup python -u main_.py --label --method updateW --GPU 1 > result/distilbert_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method block_opt --GPU 1  > result/distilbert_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --method compeft --GPU 3  > result/distilbert_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --method topk_AB --GPU 3  > result/distilbert_20news_niid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method prune --GPU 3  > result/distilbert_20news_niid_prune.out 2>&1 &



使用adamW优化器
nohup python -u main_.py --label --partition --method updateW --GPU 0 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --factor 2 --point 51 > result/updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method block_opt --GPU 1 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --blocks 2 > result/block.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --GPU 2 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/compeft.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method updateW --GPU 0 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --factor 1 --point 51 > result/updateW.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method block_opt --GPU 1 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --blocks 2 > result/block.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method compeft --GPU 2 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/compeft.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method updateW --GPU 3 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --factor 1 --point 51 > result/llupdateW.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method block_opt --GPU 4 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 --blocks 2 > result/llblock.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method compeft --GPU 5 --lr 1e-3 --max_length 512 --batch_size 16 --comm_round 50 > result/llcompeft.out 2>&1 &



nohup python -u main_.py --label --method topk_AB --dataset glue --subdataset cola --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 0 > result/cola_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --dataset glue --subdataset cola --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 1 > result/cola_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --dataset glue --subdataset cola --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 2 > result/cola_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --dataset glue --subdataset cola --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 3 > result/cola_compeft.out 2>&1 &
nohup python -u main_.py --label --method block_opt --dataset glue --subdataset cola --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 4 > result/cola_block.out 2>&1 &

nohup python -u main_.py --label --method topk_AB --lr 4e-5 --dataset glue --subdataset sst2 --model llama-3.2-1B --comm_round 10 --max_length 128 --batch_size 32 --GPU 0 > result/sst2_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --lr 4e-5 --dataset glue --subdataset sst2 --model llama-3.2-1B --comm_round 10 --max_length 128 --batch_size 32 --GPU 4 > result/sst2_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --lr 4e-5 --dataset glue --subdataset sst2 --model llama-3.2-1B --comm_round 10 --max_length 128 --batch_size 32 --GPU 5 > result/sst2_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --lr 4e-5 --dataset glue --subdataset sst2 --model llama-3.2-1B --comm_round 10 --max_length 128 --batch_size 32 --GPU 6 > result/sst2_compeft.out 2>&1 &
nohup python -u main_.py --label --method block_opt --lr 4e-5 --dataset glue --subdataset sst2 --model llama-3.2-1B --comm_round 10 --max_length 128 --batch_size 32 --GPU 7 > result/sst2_block.out 2>&1 &

nohup python -u main_.py --label --method topk_AB --dataset glue --subdataset mrpc --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 0 > result/mrpc_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --dataset glue --subdataset mrpc --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 1 > result/mrpc_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --dataset glue --subdataset mrpc --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 2 > result/mrpc_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --dataset glue --subdataset mrpc --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 3 > result/mrpc_compeft.out 2>&1 &
nohup python -u main_.py --label --method block_opt --dataset glue --subdataset mrpc --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 4 > result/mrpc_block.out 2>&1 &

nohup python -u main_.py --method topk_AB --dataset glue --subdataset stsb --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 0 > result/stsb_topk_AB.out 2>&1 &
nohup python -u main_.py --method updateW --dataset glue --subdataset stsb --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 1 > result/stsb_updateW.out 2>&1 &
nohup python -u main_.py --method prune --dataset glue --subdataset stsb --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 2 > result/stsb_prune.out 2>&1 &
nohup python -u main_.py --method compeft --dataset glue --subdataset stsb --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 3 > result/stsb_compeft.out 2>&1 &
nohup python -u main_.py --method block_opt --dataset glue --subdataset stsb --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 4 > result/stsb_block.out 2>&1 &

nohup python -u main_.py --label --method topk_AB --dataset glue --subdataset qqp --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 0 > result/qqp_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --dataset glue --subdataset qqp --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 1 > result/qqp_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --dataset glue --subdataset qqp --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 2 > result/qqp_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --dataset glue --subdataset qqp --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 3 > result/qqp_compeft.out 2>&1 &
nohup python -u main_.py --label --method block_opt --dataset glue --subdataset qqp --model llama-3.2-1B --comm_round 10 --max_length 128 --GPU 4 > result/qqp_block.out 2>&1 &

nohup python -u main_.py --label --method topk_AB --lr 1e-4 --dataset glue --subdataset mnli --model llama-3.2-1B --comm_round 10 --GPU 0 --batch_size 32 > result/mnli_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --lr 1e-4 --dataset glue --subdataset mnli --model llama-3.2-1B --comm_round 10 --GPU 1 --batch_size 32 > result/mnli_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --lr 1e-4 --dataset glue --subdataset mnli --model llama-3.2-1B --comm_round 10 --GPU 2 --batch_size 32 > result/mnli_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --lr 1e-4 --dataset glue --subdataset mnli --model llama-3.2-1B --comm_round 10 --GPU 3 --batch_size 32 > result/mnli_compeft.out 2>&1 &
nohup python -u main_.py --label --method block_opt --lr 1e-4 --dataset glue --subdataset mnli --model llama-3.2-1B --comm_round 10 --GPU 4 --batch_size 32 > result/mnli_block.out 2>&1 &


nohup python -u main_.py --label --method topk_AB --lr 1e-4 --dataset glue --subdataset qnli --model llama-3.2-1B --comm_round 10 --GPU 0 --batch_size 32 > result/qnli_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --lr 1e-4 --dataset glue --subdataset qnli --model llama-3.2-1B --comm_round 10 --GPU 1 --batch_size 32 > result/qnli_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --lr 1e-4 --dataset glue --subdataset qnli --model llama-3.2-1B --comm_round 10 --GPU 2 --batch_size 32 > result/qnli_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --lr 1e-4 --dataset glue --subdataset qnli --model llama-3.2-1B --comm_round 10 --GPU 3 --batch_size 32 > result/qnli_compeft.out 2>&1 &
nohup python -u main_.py --label --method block_opt --lr 1e-4 --dataset glue --subdataset qnli --model llama-3.2-1B --comm_round 10 --GPU 4 --batch_size 32 > result/qnli_block.out 2>&1 &

nohup python -u main_.py --method topk_AB --lr 2e-4 --dataset squad --model llama-3.2-1B --comm_round 10 --GPU 0 --batch_size 32 > result/squad_topk_AB.out 2>&1 &
nohup python -u main_.py --method updateW --lr 2e-4 --dataset squad --model llama-3.2-1B --comm_round 10 --GPU 1 --batch_size 32 > result/squad_updateW.out 2>&1 &
nohup python -u main_.py --method block_opt --lr 2e-4 --dataset squad --model llama-3.2-1B --comm_round 10 --GPU 4 --batch_size 32 > result/squad_block.out 2>&1 &
nohup python -u main_.py --method prune --lr 2e-4 --dataset squad --model llama-3.2-1B --comm_round 10 --GPU 2 --batch_size 32 > result/squad_prune.out 2>&1 &
nohup python -u main_.py --method compeft --lr 2e-4 --dataset squad --model llama-3.2-1B --comm_round 10 --GPU 3 --batch_size 32 > result/squad_compeft.out 2>&1 &



















nohup python -u main_.py --label --partition --method updateW --lr 3e-4 --GPU 2 --point 5 > result/distilbert_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --lr 3e-4 --optimize 1 --GPU 6  --point 5 > result/distilbert_20news_iid_optim.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --lr 3e-4 --GPU 0  > result/distilbert_20news_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method topk_AB --lr 3e-4 --GPU 7  > result/distilbert_20news_iid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --partition --method block_opt --lr 3e-4 --GPU 3  > result/distilbert_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --lr 3e-4 --GPU 5  > result/distilbert_20news_iid_prune.out 2>&1 &

nohup python -u main_.py --label --method updateW --lr 3e-4 --GPU 2 --point 5 > result/distilbert_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method updateW --lr 3e-4 --optimize 1 --GPU 6 --point 5  > result/distilbert_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --method compeft --lr 3e-4 --GPU 0  > result/distilbert_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --method topk_AB --lr 3e-4 --GPU 7  > result/distilbert_20news_niid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method block_opt --lr 3e-4 --GPU 3  > result/distilbert_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --method prune --lr 3e-4 --GPU 5  > result/distilbert_20news_niid_prune.out 2>&1 &


nohup python -u main_.py --label --method updateW --GPU 1 > result/distilbert_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method topk_AB --GPU 2 > result/distilbert_20news_niid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method prune --GPU 3 > result/distilbert_20news_niid_prune.out 2>&1 &
nohup python -u main_.py --label --method updateW --optimize 1 --GPU 4 > result/distilbert_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --method compeft --GPU 5 > result/distilbert_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --method block_opt --GPU 6 > result/distilbert_20news_niid_block.out 2>&1 &


nohup python -u main_.py --label --method updateW --model roberta-large --GPU 3 > result/largeroberta_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method compeft --model roberta-large --GPU 5  > result/largeroberta_20news_niid_compeft.out 2>&1 &
nohup python -u main_.py --label --method topk_AB --model roberta-large --GPU 2  > result/largeroberta_20news_niid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method block_opt --model roberta-large --GPU 6  > result/largeroberta_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --method prune --model roberta-large --GPU 1  > result/largeroberta_20news_niid_prune.out 2>&1 &
nohup python -u main_.py --label --method updateW --model roberta-large --optimize 1 --GPU 4  > result/largeroberta_20news_niid_optim.out 2>&1 &


nohup python -u main_.py --label --method updateW --model llama-2-7B --GPU 4 > result/llama_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method compeft --model llama-2-7B --GPU 7 > result/llama_20news_niid_compeft.out 2>&1 &

nohup python -u main_.py --label --method topk_AB --model llama-2-7B --GPU 1 > result/llama_20news_niid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method prune --model llama-2-7B --GPU 2 > result/llama_20news_niid_prune.out 2>&1 &
nohup python -u main_.py --label --method block_opt --model llama-2-7B --GPU 3 > result/llama_20news_niid_block.out 2>&1 &

===============================================================================================================
nohup python -u main_.py --label --partition --method block_opt > result/roberta_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --method block_opt > result/roberta_20news_niid_block.out 2>&1 &
nohup python -u main_.py --label --partition --method block_opt --model roberta-large > result/largeroberta_20news_iid_block.out 2>&1 &
nohup python -u main_.py --label --method block_opt --model roberta-large > result/largeroberta_20news_niid_block.out 2>&1 &


nohup python -u main_.py --label --partition --method updateW --model llama-2-7B --lr 1e-3 > result/llama_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method topk_AB --model llama-2-7B --lr 1e-3 > result/llama_20news_iid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --model llama-2-7B --lr 1e-3 > result/llama_20news_iid_prune.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --model llama-2-7B --lr 1e-3 --optimize 1 > result/llama_20news_iid_optim.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --model llama-2-7B --lr 1e-3 > result/llama_20news_iid_compeft.out 2>&1 &

nohup python -u main_.py --label --method updateW --model llama-2-7B > result/llama_20news_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method topk_AB --model llama-2-7B > result/llama_20news_niid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method prune --model llama-2-7B > result/llama_20news_niid_prune.out 2>&1 &
nohup python -u main_.py --label --method updateW --model llama-2-7B --optimize 1 > result/llama_20news_niid_optim.out 2>&1 &
nohup python -u main_.py --label --method compeft --model llama-2-7B > result/llama_20news_niid_compeft.out 2>&1 &


nohup python -u main_.py --label --method topk_AB --dataset glue --subdataset cola --model roberta-base --comm_round 10 > result/roberta_cola_niid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --dataset glue --subdataset cola --model roberta-base --comm_round 10 > result/roberta_cola_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --dataset glue --subdataset cola --model roberta-base --comm_round 10 > result/roberta_cola_niid_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --dataset glue --subdataset cola --model roberta-base --comm_round 10 > result/roberta_cola_niid_compeft.out 2>&1 &

nohup python -u main_.py --label --method topk_AB --dataset glue --subdataset sst2 --model roberta-base --comm_round 10 > result/roberta_sst2_niid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --method updateW --dataset glue --subdataset sst2 --model roberta-base --comm_round 10 > result/roberta_sst2_niid_updateW.out 2>&1 &
nohup python -u main_.py --label --method prune --dataset glue --subdataset sst2 --model roberta-base --comm_round 10 > result/roberta_sst2_niid_prune.out 2>&1 &
nohup python -u main_.py --label --method compeft --dataset glue --subdataset sst2 --model roberta-base --comm_round 10 > result/roberta_sst2_niid_compeft.out 2>&1 &

nohup python -u main_.py --label --partition --method topk_AB --dataset glue --subdataset qnli --model roberta-base --comm_round 100 > result/roberta_qnli_iid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --dataset glue --subdataset qnli --model roberta-base --comm_round 100 > result/roberta_qnli_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --dataset glue --subdataset qnli --model roberta-base --comm_round 100 > result/roberta_qnli_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --dataset glue --subdataset qnli --model roberta-base --comm_round 100 > result/roberta_qnli_iid_prune.out 2>&1 &

nohup python -u main_.py --label --partition --method topk_AB --dataset glue --subdataset sst2 --model roberta-base --comm_round 100 > result/roberta_sst2_iid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --dataset glue --subdataset sst2 --model roberta-base --comm_round 100 > result/roberta_sst2_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --dataset glue --subdataset sst2 --model roberta-base --comm_round 100 > result/roberta_sst2_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --dataset glue --subdataset sst2 --model roberta-base --comm_round 100 > result/roberta_sst2_iid_prune.out 2>&1 &

nohup python -u main_.py --label --partition --method topk_AB --dataset glue --subdataset mrpc --model roberta-base --comm_round 100 > result/roberta_mrpc_iid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --dataset glue --subdataset mrpc --model roberta-base --comm_round 100 > result/roberta_mrpc_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --dataset glue --subdataset mrpc --model roberta-base --comm_round 100 > result/roberta_mrpc_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --dataset glue --subdataset mrpc --model roberta-base --comm_round 100 > result/roberta_mrpc_iid_prune.out 2>&1 &

nohup python -u main_.py --label --partition --method topk_AB --dataset glue --subdataset stsb --model roberta-base --comm_round 100 > result/roberta_stsb_iid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --dataset glue --subdataset stsb --model roberta-base --comm_round 100 > result/roberta_stsb_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method compeft --dataset glue --subdataset stsb --model roberta-base --comm_round 100 > result/roberta_stsb_iid_compeft.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --dataset glue --subdataset stsb --model roberta-base --comm_round 100 > result/roberta_stsb_iid_prune.out 2>&1 &








nohup python -u main_.py --label --partition --method CGFedLLM --model roberta-base > result/roberta_20news_iid_CGFedLLM.out 2>&1 &
nohup python -u main_.py --label --partition --method STopK --model roberta-base > result/roberta_20news_iid_STopK.out 2>&1 &
nohup python -u main_.py --label --partition --method raw --model roberta-base > result/roberta_20news_iid_raw.out 2>&1 &

nohup python -u main_.py --label --partition --method STopK --model roberta-large > result/largeroberta_20news_iid_STopK.out 2>&1 &
nohup python -u main_.py --label --partition --method pq --model roberta-large > result/largeroberta_20news_iid_pq.out 2>&1 &
nohup python -u main_.py --label --partition --method prune --model roberta-large > result/largeroberta_20news_iid_prune.out 2>&1 &
nohup python -u main_.py --label --partition --method topk_AB --model roberta-large > result/largeroberta_20news_iid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --model roberta-large > result/largeroberta_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method raw --model roberta-large > result/largeroberta_20news_iid_raw.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --model roberta-large --optimize 1 > result/largeroberta_20news_iid_optim.out 2>&1 &

nohup python -u main_.py --label --partition --method updateW --model roberta-large --optimize 1 > result/largeroberta_20news_iid_test.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --model roberta-base --optimize 1 > result/roberta_20news_iid_test.out 2>&1 &

nohup python -u main_.py --label --partition --method raw --model roberta-large > result/roberta_20news_iid_raw.out 2>&1 &
nohup python -u main_.py --label --partition --method topk_AB --model roberta-large > result/roberta_20news_iid_topk_AB.out 2>&1 &
nohup python -u main_.py --label --partition --method STopK --model roberta-large > result/roberta_20news_iid_STopK.out 2>&1 &
nohup python -u main_.py --label --partition --method updateW --model roberta-large > result/roberta_20news_iid_updateW.out 2>&1 &
nohup python -u main_.py --label --partition --method new --model roberta-large > result/roberta_20news_iid_new.out 2>&1 &



nohup python -u main_.py --label > raw.out 2>&1 &
nohup python -u main_.py --label  --method topk > topk.out 2>&1 &
nohup python -u main_.py --label  --method topk_AB > topk_AB.out 2>&1 &
nohup python -u main_.py --label  --method randk_AB > randk_AB.out 2>&1 &
nohup python -u main_.py --label  --method new > new.out 2>&1 &
nohup python -u main_.py --label  --method new > 1new.out 2>&1 &
nohup python -u main_.py --label  --method new > 2new.out 2>&1 &

nohup python -u main_.py --label  --method raw > raw.out 2>&1 &
nohup python -u main_.py --label  --method STopK > l2STopK.out 2>&1 &
nohup python -u main_.py --label  --method updateW > updateW.out 2>&1 &
nohup python -u main_.py --label  --method STopK --optimize 1 > optim.out 2>&1 &
nohup python -u main_.py --label  --method topk_AB > topk_AB.out 2>&1 &

nohup python -u main_.py --label  --method STopK --model llama-3.2-1B > STopK_1B.out 2>&1 &
nohup python -u main_.py --label  --method updateW --model llama-3.2-1B > updateW_1B.out 2>&1 &


nohup python -u main_.py --label  --method raw --model roberta-large > raw_03B.out 2>&1 &
nohup python -u main_.py --label  --method STopK --model roberta-large --test_bs 16 > STopK_03B.out 2>&1 &
nohup python -u main_.py --label  --method updateW --model roberta-large --test_bs 16 > updateW_03B.out 2>&1 &
nohup python -u main_.py --label  --method STopK --optimize 1 --model roberta-large > optim_03B.out 2>&1 &
nohup python -u main_.py --label  --method topk_AB --model roberta-large > topk_AB_03B.out 2>&1 &

nohup python -u main_.py --label  --method topk_AB --model llama-2-7B --batch_size 8 --test_bs 8 --comm_round 10 > topk_AB_7B.out 2>&1 &
nohup python -u main_.py --label  --method STopK --model llama-2-7B --batch_size 8 --test_bs 8 --comm_round 10 > l2STopK_7B.out 2>&1 &
nohup python -u main_.py --label  --method updateW --model llama-2-7B --batch_size 8 --test_bs 8 --comm_round 10 > updateW_7B.out 2>&1 &

new.out是使用设计的算法进行求解
1new.out\2new.out是使用遍历的方法来求解最大值和最小值最优比例的方法进行求解，1是使用放大的损失函数，2是使用真实的损失函数

nohup python -u main_.py --label --comm_round 100  --method topk_AB > 100topk_AB.out 2>&1 &
nohup python -u main_.py --label --comm_round 100  --method randk_AB > 100randk_AB.out 2>&1 &
nohup python -u main_.py --label --comm_round 100  --method new > 100new.out 2>&1 &


nohup python -u main_.py --label --model distilbert-base-multilingual-cased > result/distilbert-base-multilingual-cased/raw.out 2>&1 &
nohup python -u main_.py --label  --method topk --model distilbert-base-multilingual-cased > result/distilbert-base-multilingual-cased/topk.out 2>&1 &
nohup python -u main_.py --label  --method topk_AB --model distilbert-base-multilingual-cased > result/distilbert-base-multilingual-cased/topk_AB.out 2>&1 &

nohup python -u main_.py --label --model roberta-large > result/robert_large/raw.out 2>&1 &
nohup python -u main_.py --label  --method topk --model roberta-large > result/robert_large/topk.out 2>&1 &
nohup python -u main_.py --label  --method topk_AB --model roberta-large > result/robert_large/topk_AB.out 2>&1 &

nohup python -u main_.py --label  --model llama-3.2-1B > result/llama-3.2-1B/raw.out 2>&1 &
nohup python -u main_.py --label  --method topk --model llama-3.2-1B > result/llama-3.2-1B/topk.out 2>&1 &
nohup python -u main_.py --label  --method topk_AB --model llama-3.2-1B > result/llama-3.2-1B/topk_AB.out 2>&1 &
