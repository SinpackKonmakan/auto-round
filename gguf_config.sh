for model_name in "Qwen3-8B"; do
device=2
format=gguf:q5_k_s
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --iters 200 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/ \
        --eval_bs 16 \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}.log
done