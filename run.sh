# 统计运行时间
python timing_analyzer.py --model roberta-large --dataset 20_newsgroups --max_length 256 --batch_size 32 --lr 1e-3 

nohup python -u main_.py --label --partition --model roberta-large --method raw --GPU 0 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-3 --momentum 0.9 > result/roberta_20news_iid_raw.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method HeLoRA --GPU 4 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-3 --momentum 0.9 > result/roberta_20news_iid_helora.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method pq --GPU 3 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-3 --momentum 0.9 > result/roberta_20news_iid_pq.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method topk --GPU 7 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-3 --momentum 0.9 > result/roberta_20news_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method FFTHM --GPU 6 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-3 --momentum 0.9 > result/roberta_20news_iid_ffthm.out 2>&1 &

nohup python -u main_.py --label --model roberta-large --method raw --GPU 0 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-3 --momentum 0.9 > result/roberta_20news_niid_raw.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method HeLoRA --GPU 4 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-3 --momentum 0.9 > result/roberta_20news_niid_helora.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method pq --GPU 3 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-3 --momentum 0.9 > result/roberta_20news_niid_pq.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method topk --GPU 7 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-3 --momentum 0.9 > result/roberta_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --model roberta-large --method FFTHM --GPU 6 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-3 --momentum 0.9 > result/roberta_20news_niid_ffthm.out 2>&1 &



python timing_analyzer.py --model distilbert-base-multilingual-cased --dataset 20_newsgroups --max_length 512 --batch_size 32 --lr 1e-3 

nohup python -u main_.py --label --partition --model distilbert-base-multilingual-cased --method raw --GPU 0 --max_length 512 --batch_size 16 --comm_round 75 --lr 1e-3 --momentum 0.9 > result/distilbert_20news_iid_raw.out 2>&1 &
nohup python -u main_.py --label --partition --model distilbert-base-multilingual-cased --method HeLoRA --GPU 4 --max_length 512 --batch_size 16 --comm_round 75 --lr 1e-3 --momentum 0.9 > result/distilbert_20news_iid_helora.out 2>&1 &
nohup python -u main_.py --label --partition --model distilbert-base-multilingual-cased --method pq --GPU 3 --max_length 512 --batch_size 16 --comm_round 75 --lr 1e-3 --momentum 0.9 > result/distilbert_20news_iid_pq.out 2>&1 &
nohup python -u main_.py --label --partition --model distilbert-base-multilingual-cased --method topk --GPU 7 --max_length 512 --batch_size 16 --comm_round 75 --lr 1e-3 --momentum 0.9 > result/distilbert_20news_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --model distilbert-base-multilingual-cased --method FFTHM --GPU 6 --max_length 512 --batch_size 16 --comm_round 75 --lr 1e-3 --momentum 0.9 > result/distilbert_20news_iid_ffthm.out 2>&1 &

nohup python -u main_.py --label --model distilbert-base-multilingual-cased --method raw --GPU 0 --max_length 512 --batch_size 16 --comm_round 75 --lr 1e-3 --momentum 0.9 > result/distilbert_20news_niid_raw.out 2>&1 &
nohup python -u main_.py --label --model distilbert-base-multilingual-cased --method HeLoRA --GPU 4 --max_length 512 --batch_size 16 --comm_round 75 --lr 1e-3 --momentum 0.9 > result/distilbert_20news_niid_helora.out 2>&1 &
nohup python -u main_.py --label --model distilbert-base-multilingual-cased --method pq --GPU 3 --max_length 512 --batch_size 16 --comm_round 75 --lr 1e-3 --momentum 0.9 > result/distilbert_20news_niid_pq.out 2>&1 &
nohup python -u main_.py --label --model distilbert-base-multilingual-cased --method topk --GPU 7 --max_length 512 --batch_size 16 --comm_round 75 --lr 1e-3 --momentum 0.9 > result/distilbert_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --model distilbert-base-multilingual-cased --method FFTHM --GPU 6 --max_length 512 --batch_size 16 --comm_round 75 --lr 1e-3 --momentum 0.9 > result/distilbert_20news_niid_ffthm.out 2>&1 &



python timing_analyzer.py --model llama-3.2-1B --dataset 20_newsgroups --max_length 256 --batch_size 64 --lr 1e-3 

nohup python -u main_.py --label --partition --model llama-3.2-1B --method raw --GPU 0 --max_length 128 --batch_size 128 --comm_round 20 --lr 1e-3 --momentum 0.9 > result/llama_20news_iid_raw.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method HeLoRA --GPU 1 --max_length 128 --batch_size 128 --comm_round 20 --lr 1e-3 --momentum 0.9 > result/llama_20news_iid_helora.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method pq --GPU 2 --max_length 128 --batch_size 128 --comm_round 20 --lr 1e-3 --momentum 0.9 > result/llama_20news_iid_pq.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method topk --GPU 3 --max_length 128 --batch_size 128 --comm_round 20 --lr 1e-3 --momentum 0.9 > result/llama_20news_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method FFTHM --GPU 0 --max_length 128 --batch_size 128 --comm_round 20 --lr 1e-3 --momentum 0.9 > result/llama_20news_iid_ffthm.out 2>&1 &

nohup python -u main_.py --label --model llama-3.2-1B --method raw --GPU 4 --max_length 128 --batch_size 128 --comm_round 20 --lr 1e-3 --momentum 0.9 > result/llama_20news_niid_raw.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method HeLoRA --GPU 5 --max_length 128 --batch_size 128 --comm_round 20 --lr 1e-3 --momentum 0.9 > result/llama_20news_niid_helora.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method pq --GPU 6 --max_length 128 --batch_size 128 --comm_round 20 --lr 1e-3 --momentum 0.9 > result/llama_20news_niid_pq.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method topk --GPU 7 --max_length 128 --batch_size 128 --comm_round 20 --lr 1e-3 --momentum 0.9 > result/llama_20news_niid_topk.out 2>&1 &
nohup python -u main_.py --label --model llama-3.2-1B --method FFTHM --GPU 1 --max_length 128 --batch_size 128 --comm_round 20 --lr 1e-3 --momentum 0.9 > result/llama_20news_niid_ffthm.out 2>&1 &





python timing_analyzer.py --model roberta-large --dataset squad --max_length 128 --batch_size 32 --lr 1e-2 

nohup python -u main_.py --label --partition --model roberta-large --method raw --GPU 1 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/roberta_squad_iid_raw.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method HeLoRA --GPU 4 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/roberta_squad_iid_helora.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method pq --GPU 5 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/roberta_squad_iid_pq.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method topk --GPU 7 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/roberta_squad_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --model roberta-large --method FFTHM --GPU 2 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/roberta_squad_iid_ffthm.out 2>&1 &

nohup python -u main_.py --model roberta-large --method raw --GPU 0 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/roberta_squad_niid_raw.out 2>&1 &
nohup python -u main_.py --model roberta-large --method HeLoRA --GPU 1 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/roberta_squad_niid_helora.out 2>&1 &
nohup python -u main_.py --model roberta-large --method pq --GPU 2 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/roberta_squad_niid_pq.out 2>&1 &
nohup python -u main_.py --model roberta-large --method topk --GPU 1 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/roberta_squad_niid_topk.out 2>&1 &
nohup python -u main_.py --model roberta-large --method FFTHM --GPU 2 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/roberta_squad_niid_ffthm.out 2>&1 &


python timing_analyzer.py --model distilbert-base-multilingual-cased --dataset squad --max_length 512 --batch_size 32 --lr 1e-3 

nohup python -u main_.py --model distilbert-base-multilingual-cased --label --partition --method raw --GPU 0 --max_length 256 --batch_size 64 --comm_round 30 --lr 1e-1 --momentum 0.9 --dataset squad > result/distilbert_squad_iid_raw.out 2>&1 &
nohup python -u main_.py --model distilbert-base-multilingual-cased --label --partition --method HeLoRA --GPU 2 --max_length 256 --batch_size 64 --comm_round 30 --lr 1e-1 --momentum 0.9 --dataset squad > result/distilbert_squad_iid_helora.out 2>&1 &
nohup python -u main_.py --model distilbert-base-multilingual-cased --label --partition --method pq --GPU 4 --max_length 256 --batch_size 64 --comm_round 30 --lr 1e-1 --momentum 0.9 --dataset squad > result/distilbert_squad_iid_pq.out 2>&1 &
nohup python -u main_.py --model distilbert-base-multilingual-cased --label --partition --method topk --GPU 6 --max_length 256 --batch_size 64 --comm_round 30 --lr 1e-1 --momentum 0.9 --dataset squad > result/distilbert_squad_iid_topk.out 2>&1 &
nohup python -u main_.py --model distilbert-base-multilingual-cased --label --partition --method FFTHM --GPU 3 --max_length 256 --batch_size 64 --comm_round 30 --lr 1e-1 --momentum 0.9 --dataset squad > result/distilbert_squad_iid_ffthm.out 2>&1 &

nohup python -u main_.py --model distilbert-base-multilingual-cased --method raw --GPU 0 --max_length 256 --batch_size 64 --comm_round 30 --lr 1e-1 --momentum 0.9 --dataset squad > result/distilbert_squad_niid_raw.out 2>&1 &
nohup python -u main_.py --model distilbert-base-multilingual-cased --method HeLoRA --GPU 2 --max_length 256 --batch_size 64 --comm_round 30 --lr 1e-1 --momentum 0.9 --dataset squad > result/distilbert_squad_niid_helora.out 2>&1 &
nohup python -u main_.py --model distilbert-base-multilingual-cased --method pq --GPU 4 --max_length 256 --batch_size 64 --comm_round 30 --lr 1e-1 --momentum 0.9 --dataset squad > result/distilbert_squad_niid_pq.out 2>&1 &
nohup python -u main_.py --model distilbert-base-multilingual-cased --method topk --GPU 6 --max_length 256 --batch_size 64 --comm_round 30 --lr 1e-1 --momentum 0.9 --dataset squad > result/distilbert_squad_niid_topk.out 2>&1 &
nohup python -u main_.py --model distilbert-base-multilingual-cased --method FFTHM --GPU 3 --max_length 256 --batch_size 64 --comm_round 30 --lr 1e-1 --momentum 0.9 --dataset squad > result/distilbert_squad_niid_ffthm.out 2>&1 &



python timing_analyzer.py --model llama-3.2-1B --dataset squad --max_length 128 --batch_size 128 --lr 1e-3 

nohup python -u main_.py --label --partition --model llama-3.2-1B --method raw --GPU 2 --max_length 128 --batch_size 128 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/llama_squad_iid_raw.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method HeLoRA --GPU 3 --max_length 128 --batch_size 128 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/llama_squad_iid_helora.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method pq --GPU 4 --max_length 128 --batch_size 128 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/llama_squad_iid_pq.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method topk --GPU 5 --max_length 128 --batch_size 128 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/llama_squad_iid_topk.out 2>&1 &
nohup python -u main_.py --label --partition --model llama-3.2-1B --method FFTHM --GPU 4 --max_length 128 --batch_size 128 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/llama_squad_iid_ffthm.out 2>&1 &

nohup python -u main_.py --model llama-3.2-1B --method raw --GPU 0 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/llama_squad_niid_raw.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method HeLoRA --GPU 1 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/llama_squad_niid_helora.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method pq --GPU 2 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/llama_squad_niid_pq.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method topk --GPU 3 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/llama_squad_niid_topk.out 2>&1 &
nohup python -u main_.py --model llama-3.2-1B --method FFTHM --GPU 7 --max_length 128 --batch_size 32 --comm_round 30 --lr 1e-2 --momentum 0.9 --dataset squad > result/llama_squad_niid_ffthm.out 2>&1 &
