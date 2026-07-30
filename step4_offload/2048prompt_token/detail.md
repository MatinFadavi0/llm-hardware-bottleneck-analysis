model name : Qwen/Qwen2.5-1.5B-Instruct
vram-utilization : 0.53
max-model-len : 2500

serve and evaluate model with  :  

nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --cpuctxsw=none \
  --backtrace=none \
  --cuda-graph-trace=node \
  --output=qwen_profile \
  --force-overwrite=true \
  vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --gpu-memory-utilization 0.53 \
    --max-model-len 2500 \
    --port 8000 \
    --cpu-offload-gb 2.5 \
    --enforce-eager

use guidllm with :

 guidellm run \
    --backend kind=openai_http,target="http://localhost:8000/v1",model="Qwen/Qwen2.5-1.5B-Instruct" \
    --data kind=synthetic_text,prompt_tokens=2048,output_tokens=128 \
    --profile kind=constant,rate=5 \
    --constraint kind=max_requests,count=50 \
    --output kind=json,path="./results_step4.json"


get the csv file with : nsys stats --report cuda_gpu_kern_sum --format csv --output kernel_report qwen_profile.nsys-rep