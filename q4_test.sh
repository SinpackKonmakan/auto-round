for model_name in "Qwen3-8B"; do
device=3
format=gguf:q4_k_m
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --data_type int_asym_dq \
        --group_size 32 \
        --super_bits 6 \
        --super_group_size 8 \
        --bits 4 \
        --iters 0 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/${format}_${model_name}_search \
        --eval_bs 16 \
        --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu,leaderboard_ifeval,leaderboard_mmlu_pro,gsm8k \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}_search.log
done