for model_name in "Qwen2.5-7B-Instruct" "Qwen3-8B " "Meta-Llama-3.1-8B-Instruct" "phi-4" "falcon-three-7b"; do
device=2
format="gguf:q4_k_s,fake"
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --iters 0 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/ \
        --eval_bs 16 \
        --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu,leaderboard_ifeval,leaderboard_mmlu_pro,gsm8k \
        2>&1 | tee /data5/shiqi/log/${format}_${model_name}_autoround.log
done