models="Qwen2.5-7B-Instruct" "falcon-three-7b" "Meta-Llama-3.1-8B-Instruct" "phi-4"
for model_name in "Qwen2.5-7B-Instruct" "falcon-three-7b" "Meta-Llama-3.1-8B-Instruct" "phi-4"; do
device=2
format=fake
CUDA_VISIBLE_DEVICES=$device python -m auto_round \
        --format ${format} \
        --data_type int_asym_dq \
        --group_size 16 \
        --super_bits 4 \
        --act_bits 16 \
        --super_group_size 16 \
        --bits 2 \
        --iters 200 \
        --asym \
        --rrmin -0.5 \
        --rdelta 0.1 \
        --nstep 15 \
        --disable_minmax_tuning \
        --model /models/${model_name} \
        --output_dir /data5/shiqi/model/q2_${format}_${model_name}_search_tune_10time_dis \
        --eval_bs 16 \
        --tasks lambada_openai,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,piqa,truthfulqa_mc1,winogrande \
        2>&1 | tee /data5/shiqi/log/q2_${format}_${model_name}_search_tune_10time_dis.log
done