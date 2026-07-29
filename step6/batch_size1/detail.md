model name : Qwen/Qwen2.5-1.5B-Instruct
vram-utilization : 0.52
max-model-len : 2048

serve model with :  vllm serve Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.52 --max-num-seqs 16 --max-model-len 2048 --port 8000

use guidllm with : guidellm run \
    --backend kind=openai_http,target="http://localhost:8000/v1",model="Qwen/Qwen2.5-1.5B-Instruct" \
    --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
    --profile kind=constant,rate=1 \
    --constraint kind=max_requests,count=20 \
    --output kind=json,path="./guidellm_results.json"

    