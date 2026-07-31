model name : Qwen/Qwen2.5-1.5B-Instruct
vram-utilization : 0.7
max-model-len : 2500

serve and evaluate model with :  

VLLM_USE_V2_MODEL_RUNNER=0 nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --cpuctxsw=none \
  --backtrace=none \
  --cuda-graph-trace=node \
  --output=qwen_profile \
  --force-overwrite=true \
  vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --gpu-memory-utilization 0.70 \
    --max-model-len 2500 \
    --cpu-offload-gb 2 \
    --port 8000


use guidllm with : 

 guidellm run \
  --backend kind=openai_http,target="http://localhost:8000/v1",model="Qwen/Qwen2.5-1.5B-Instruct" \
  --data '{"kind":"synthetic_text","prompt_tokens":128,"output_tokens":64,"prefix_buckets":[{"bucket_weight":100,"prefix_count":2,"prefix_tokens":2000}]}' \
  --profile kind=concurrent,streams=4\
  --constraint kind=max_duration,seconds=60 \
  --output kind=json,path="./guidellm_results.json"

    

    get the csv file with : nsys stats --report cuda_gpu_kern_sum --format csv --output kernel_report qwen_profile.nsys-rep