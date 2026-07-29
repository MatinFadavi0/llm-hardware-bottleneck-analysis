profiling :

 nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=cuda_baseline_profile \
  ./step2_vectorAdd/test_vectorAdd



nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --force-export=true \
  vectoradd_profile.nsys-rep > cuda_step2_stats.txt

  
nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv \
  --output cuda_step2_report \
  --force-export=true \
  vectoradd_profile.nsys-rep


