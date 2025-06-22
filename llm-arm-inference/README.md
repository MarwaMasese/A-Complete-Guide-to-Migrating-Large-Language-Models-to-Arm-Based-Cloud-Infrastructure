# LLM ARM Inference API

A production-grade containerized LLM inference API optimized for ARM-based cloud infrastructure (AWS Graviton, Google Cloud Tau T2A, Azure Ampere Altra).

## Features

- 🚀 **ARM-optimized inference** with llama.cpp and NEON support
- 🔧 **4-bit quantization** using bitsandbytes for ARM efficiency
- 🐳 **Multi-architecture Docker** support (amd64, arm64)
- ☸️ **Kubernetes-ready** with ARM node selectors
- 📊 **Production monitoring** with health checks and metrics
- 🔄 **Auto-scaling** with Horizontal Pod Autoscaler
- 🛡️ **Security-hardened** containers with non-root users

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Transformers  │    │   llama.cpp     │
│   Web Server    │───▶│   Model Loader  │───▶│   ARM Optimized │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   REST API      │    │   4-bit Quant   │    │   NEON SIMD     │
│   /generate     │    │   (ARM only)    │    │   Instructions  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker with buildx support
- Kubernetes cluster with ARM64 nodes
- 4GB+ RAM per instance

### Local Development

1. **Clone and setup:**
```bash
git clone <repository>
cd llm-arm-inference
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run locally:**
```bash
python app/server.py
```

4. **Test the API:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_length": 100}'
```

### Docker Deployment

1. **Build multi-architecture image:**
```bash
chmod +x scripts/build_push.sh
REGISTRY=your-registry.com IMAGE_NAME=llm-arm-inference ./scripts/build_push.sh
```

2. **Run container:**
```bash
docker run -d \
  --name llm-inference \
  -p 8000:8000 \
  -e MODEL_NAME="microsoft/DialoGPT-medium" \
  your-registry.com/llm-arm-inference:latest
```

### Kubernetes Deployment

1. **Update the image in deployment:**
```bash
# Edit kubernetes/deployment-arm.yaml
# Change: your-registry.com/llm-arm-inference:latest
```

2. **Deploy to cluster:**
```bash
kubectl apply -f kubernetes/deployment-arm.yaml
```

3. **Check deployment:**
```bash
kubectl get pods -n llm-inference
kubectl logs -f deployment/llm-arm-inference -n llm-inference
```

## API Endpoints

### Core Endpoints

- `POST /generate` - Generate text from prompt
- `POST /batch_generate` - Batch text generation
- `GET /health` - Health check and system info
- `GET /model/info` - Model information
- `GET /model/stats` - Model statistics
- `GET /metrics` - Prometheus metrics

### Example Usage

**Single Generation:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_length": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50
  }'
```

**Batch Generation:**
```bash
curl -X POST "http://localhost:8000/batch_generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Hello", "How are you?", "Tell me a story"],
    "max_length": 128
  }'
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `microsoft/DialoGPT-medium` | HuggingFace model name |
| `GGML_MODEL_PATH` | `None` | Path to GGML/GGUF model file |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `WORKERS` | `1` | Number of workers |
| `HF_TOKEN` | `None` | HuggingFace API token |

### ARM Optimizations

The API automatically detects ARM architecture and applies optimizations:

- **llama.cpp with NEON**: Faster inference on ARM CPUs
- **4-bit quantization**: Reduced memory usage on ARM
- **Thread optimization**: Optimal thread count for ARM cores
- **Memory management**: ARM-specific memory optimizations

## Performance Tuning

### AWS Graviton Recommendations

```yaml
# Kubernetes resource requests for Graviton instances
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

# Node selector for Graviton instances
nodeSelector:
  kubernetes.io/arch: arm64
  node.kubernetes.io/instance-type: "m6g.large"
```

### Model Selection

| Model Size | Memory | Recommended Instance |
|------------|--------|---------------------|
| Small (1B) | 2-4GB | m6g.medium |
| Medium (7B) | 8-16GB | m6g.xlarge |
| Large (13B) | 16-32GB | m6g.2xlarge |

## Monitoring

### Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Detailed metrics
curl http://localhost:8000/metrics
```

### Prometheus Integration

The API exposes metrics at `/metrics` endpoint:

- `llm_memory_usage_bytes` - Memory usage
- `llm_cpu_usage_percent` - CPU usage
- `llm_model_loaded` - Model load status

### Grafana Dashboard

Example queries:
```promql
# Memory usage
llm_memory_usage_bytes / 1024 / 1024 / 1024

# CPU usage
llm_cpu_usage_percent

# Request rate
rate(http_requests_total[5m])
```

## Security

### Container Security

- Non-root user execution
- Read-only root filesystem
- Dropped capabilities
- Security contexts applied

### Network Security

```yaml
# Network policies (example)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llm-network-policy
spec:
  podSelector:
    matchLabels:
      app: llm-arm-inference
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
```

## Troubleshooting

### Common Issues

1. **Model loading fails:**
   ```bash
   # Check logs
   kubectl logs deployment/llm-arm-inference -n llm-inference
   
   # Verify memory limits
   kubectl describe pod <pod-name> -n llm-inference
   ```

2. **ARM detection issues:**
   ```bash
   # Check architecture
   kubectl exec -it <pod-name> -n llm-inference -- uname -m
   
   # Verify node labels
   kubectl get nodes --show-labels
   ```

3. **Performance issues:**
   ```bash
   # Check resource usage
   kubectl top pods -n llm-inference
   
   # Monitor metrics
   curl http://<service-ip>/metrics
   ```

### Debug Mode

Enable debug logging:
```bash
docker run -e LOG_LEVEL=DEBUG your-registry.com/llm-arm-inference:latest
```

## Development

### Project Structure

```
llm-arm-inference/
├── app/
│   ├── model_loader.py    # Model loading with ARM detection
│   ├── inference.py       # Inference engine
│   └── server.py          # FastAPI server
├── kubernetes/
│   └── deployment-arm.yaml # K8s deployment for ARM
├── scripts/
│   └── build_push.sh      # Multi-arch build script
├── Dockerfile             # Multi-stage ARM-optimized
├── requirements.txt       # Python dependencies
└── .dockerignore         # Docker ignore rules
```

### Adding New Models

1. **Update model_loader.py:**
```python
# Add new model configuration
MODEL_CONFIGS = {
    "your-model": {
        "name": "your-org/your-model",
        "quantization": True,
        "context_length": 2048
    }
}
```

2. **Test with new model:**
```bash
MODEL_NAME="your-org/your-model" python app/server.py
```

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure ARM compatibility
5. Submit pull request

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: Create GitHub issue
- **Discussions**: GitHub Discussions
- **Documentation**: See `/docs` directory

## Related Projects

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - ARM-optimized inference
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - Quantization
- [Transformers](https://github.com/huggingface/transformers) - Model library

---

**Built for ARM-powered cloud infrastructure** 🚀
