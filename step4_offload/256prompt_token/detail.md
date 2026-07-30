model name : Qwen/Qwen2.5-1.5B-Instruct
vram-utilization : 0.52
max-model-len : 2048

serve model with :  vllm serve Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.52 --max-model-len 2048 --port 8000 --cpu-offload-gb 1

use guidllm with : guidellm run \
    --backend kind=openai_http,target="http://localhost:8000/v1",model="Qwen/Qwen2.5-1.5B-Instruct" \
    --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
    --profile kind=constant,rate=5 \
    --constraint kind=max_requests,count=30 \
    --output kind=json,path="./results_step4.json"

    