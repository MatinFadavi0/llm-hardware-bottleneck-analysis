guidellm run \
  --backend kind=openai_http,target="http://localhost:8000/v1",model="Qwen/Qwen2.5-3B-Instruct" \
  --data '{"kind":"synthetic_text","prompt_tokens":1024,"output_tokens":128,"prefix_buckets":[{"bucket_weight":100,"prefix_count":4,"prefix_tokens":800}]}' \
  --profile kind=constant,rate=4 \
  --constraint kind=max_duration,seconds=60 \
  --output kind=json,path="./results_step4_offload.json"



    vllm serve Qwen/Qwen2.5-3B-Instruct \
    --gpu-memory-utilization 0.88 \
    --max-model-len 2048 \
    --kv-offloading-backend native \
    --kv-offloading-size 4 \
    --port 8000 \
    --enforce-eager


    vllm serve Qwen/Qwen2.5-3B-Instruct \
    --gpu-memory-utilization 0.88 \
    --max-model-len 2048 \
    --port 8000 \
    --enforce-eager