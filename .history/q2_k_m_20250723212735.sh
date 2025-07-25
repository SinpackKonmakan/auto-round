for model_name in "falcon-three-7b"; do
device=4
format=gguf:q2_k_m,fake
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --iters 200 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/models/${model_name}_iter200 \
        --eval_bs 16 \
        --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}_iter200.log
done