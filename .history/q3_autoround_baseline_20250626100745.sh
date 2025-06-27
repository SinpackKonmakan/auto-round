models="Qwen2.5-7B-Instruct" "Qwen3-8B " "falcon-three-7b" "Meta-Llama-3.1-8B-Instruct" "phi-4"
device=6
format="gguf:q3_k_s,fake"
for model_name in "Qwen2.5-7B-Instruct" "Qwen3-8B " "falcon-three-7b" "Meta-Llama-3.1-8B-Instruct" "phi-4"; do
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --data_type int_sym_dq \
        --group_size 16 \
        --super_bits 6 \
        --act_bits 16 \
        --super_group_size 16 \
        --bits 3 \
        --iters 0 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/model/ \
        --eval_bs 16 \
        --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu,leaderboard_ifeval,leaderboard_mmlu_pro,gsm8k \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}_baseline_autoround.log
done