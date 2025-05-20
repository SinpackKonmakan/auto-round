for model_name in "Qwen2.5-7B-Instruct" "falcon-three-7b" "Meta-Llama-3.1-8B-Instruct" "phi-4"; do
device=6
format=fake
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --data_type int_asym_dq \
        --group_size 32 \
        --super_bits 6 \
        --super_group_size 8 \
        --bits 4 \
        --iters 200 \
        --rrmin -1 \
        --rdelta 0.1 \
        --nstep 1 \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/model/q4_${format}_${model_name}_search_tune_accel \
        --eval_bs 16 \
        --tasks arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,mmlu,openbookqa,piqa,truthfulqa_mc1,winogrande \
        2>&1 | tee /data5/shiqi/log/q4_${format}_${model_name}_search_tune_accel.log
done