#!/usr/bin/env bash
: 'for i in 'twitter2015' # 'conll2003' 'twitter2015' 'twitter2017'--bertlayer
do
    for k in 'MTCCMBert'
    do
      echo 'run_mtmner_crf.py'
      echo ${i}
      echo ${k}
      PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python3 run_mtmner_crf.py --data_dir=data/twitter2015 --bert_model=bert-base-cased --task_name=${i} --output_dir=out/${i}_${k}_mtmner_crf_output/ --max_seq_length=128 --do_train --do_eval --train_batch_size=32 --mm_model ${k} --layer_num1=1 --layer_num2=1 --layer_num3=1
      # --do_eval --mm_model ${k} --layer_num1=1 --layer_num2=1 --layer_num3=1
    done
done
'
python3 run_mtmner_crf.py --data_dir=data/twitter2015 --bert_model=bert-base-cased --task_name=twitter2015 --output_dir=out/mtmner_crf_output/ --local_rank=0 --max_seq_length=128 --do_train --do_eval --train_batch_size=32 --mm_model MTCCMBert --layer_num1=1 --layer_num2=1 --layer_num3=1