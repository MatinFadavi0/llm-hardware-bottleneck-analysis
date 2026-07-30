model name : Qwen/Qwen2.5-1.5B-Instruct
vram-utilization : 0.53
max-model-len : 2500

serve model with :  vllm serve Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.53 --max-model-len 2500 --port 8000

use guidllm with : guidellm run \
    --backend kind=openai_http,target="http://localhost:8000/v1",model="Qwen/Qwen2.5-1.5B-Instruct" \
    --data kind=synthetic_text,prompt_tokens=2048,output_tokens=128 \
    --profile kind=constant,rate=5 \
    --constraint kind=max_requests,count=50 \
    --output kind=json,path="./results_step3.json"

    