for model_name in "Qwen3-8B "; do
device=0
format="gguf:q2_k_s,fake"
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --iters 0 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/ \
        --eval_bs 16 \
        --tasks mmlu \
        2>&1 | tee /data5/shiqi/log/${format}_Qwen38B_autoround.log
done