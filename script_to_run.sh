NUM_PARTICLES=(32 16 8 4 2 1 64 128)

for P in ${NUM_PARTICLES[@]}; do
HF_TOKEN=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  python /new_data/probabilistic_inference_scaling/probabilistic_inference_scaling/scripts/pg.py \
        --total-timesteps 1 \
        --n-particles $P \
        --dataset-start 0 \
        --dataset-end 500 \
        --prm-path Qwen/Qwen2.5-Math-PRM-7B \
        --softmax-temp 1 \
        --seed 96 \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --output-dir /new_data/probabilistic_inference_scaling/probabilistic_inference_scaling/llama8b_qwenRM_results_jan20/seed96/softmax_temp1/model_tempPoint8/p$P/ \
	--resample-inactive
done

#YOU ONLY HAVE TO CHANGE THE LOCATION OF PG.PY AND THE OUTPUT DIR.