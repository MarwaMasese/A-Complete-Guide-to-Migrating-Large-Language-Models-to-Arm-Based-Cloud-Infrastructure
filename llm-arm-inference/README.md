# LLM ARM Inference API

A production-grade containerized LLM inference API optimized for ARM-based cloud infrastructure (AWS Graviton, Google Cloud Tau T2A, Azure Ampere Altra).

## 🚀 Features

- **Multi-Architecture Support**: Works on both x86_64 and ARM64 platforms
- **ARM-Optimized Inference**: Automatic platform detection with ARM-specific optimizations
- **4-bit Quantization**: Memory-efficient inference using bitsandbytes (ARM only)
- **Production-Ready**: FastAPI with health checks, metrics, and monitoring
- **Container-Native**: Multi-architecture Docker support (amd64, arm64)
- **Kubernetes-Ready**: ARM node selectors and auto-scaling
- **Security-Hardened**: Non-root containers with security contexts

## 📋 Quick Start

### Prerequisites

- Python 3.11+
- 4GB+ RAM
- Docker (optional)
- Kubernetes cluster (optional)

### Local Setup

1. **Clone the repository:**
```bash
git clone <repository>
cd llm-arm-inference
```

2. **Set up Python environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Start the server:**
```bash
python app/server.py
```

5. **Test the API:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_length": 100}'
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t llm-arm-inference .

# Run container
docker run -d \
  --name llm-inference \
  -p 8000:8000 \
  -e MODEL_NAME="microsoft/DialoGPT-medium" \
  llm-arm-inference
```

### Multi-Architecture Build

```bash
# Enable buildx
docker buildx create --use

# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag your-registry.com/llm-arm-inference:latest \
  --push .
```

## ☸️ Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/deployment-arm.yaml

# Check status
kubectl get pods -n llm-inference
kubectl logs -f deployment/llm-arm-inference -n llm-inference
```

## 📚 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate` | Generate text from prompt |
| `POST` | `/batch_generate` | Batch text generation |
| `GET` | `/health` | Health check and system info |
| `GET` | `/model/info` | Model information |
| `GET` | `/model/stats` | Model statistics |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/` | API information |

### Generate Text

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_length": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
  }'
```

**Response:**
```json
{
  "response": "The future of AI is bright and full of possibilities...",
  "prompt": "The future of AI is",
  "generation_time": 2.34,
  "model_info": {
    "model_name": "microsoft/DialoGPT-medium",
    "architecture": "x86_64",
    "device": "cpu",
    "quantized": false,
    "loaded": true
  }
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "system_info": {
    "platform": "Windows-10-10.0.19045-SP0",
    "architecture": "AMD64",
    "python_version": "3.11.0",
    "cpu_count": 8,
    "memory_total_gb": 16.0,
    "memory_available_gb": 8.5,
    "cpu_usage_percent": 15.2
  },
  "model_info": {
    "model_name": "microsoft/DialoGPT-medium",
    "architecture": "x86_64",
    "device": "cpu",
    "quantized": false,
    "loaded": true
  }
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `microsoft/DialoGPT-medium` | HuggingFace model name |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `WORKERS` | `1` | Number of workers |
| `HF_TOKEN` | - | HuggingFace API token (optional) |
| `LOG_LEVEL` | `INFO` | Logging level |

### Model Configuration

```python
# Supported models (examples)
MODEL_NAME="microsoft/DialoGPT-medium"    # Conversational AI
MODEL_NAME="gpt2"                         # Text generation
MODEL_NAME="facebook/opt-350m"            # Lightweight option
MODEL_NAME="EleutherAI/gpt-neo-125M"     # Small GPT model
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Model Loader  │    │   Inference     │
│   Web Server    │───▶│   Platform      │───▶│   Engine        │
│                 │    │   Detection     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   REST API      │    │   ARM/x86       │    │   Transformers  │
│   Endpoints     │    │   Optimization  │    │   Backend       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Platform Detection

The API automatically detects your platform and applies optimizations:

- **x86_64**: Standard PyTorch inference
- **ARM64**: 4-bit quantization + optimized threading
- **CUDA**: GPU acceleration (if available)
- **Apple Silicon**: MPS backend support

## 📊 Monitoring

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

Available metrics:
- `llm_memory_usage_bytes` - Current memory usage
- `llm_cpu_usage_percent` - CPU utilization
- `llm_model_loaded` - Model load status (0/1)

### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'llm-inference'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

## 🚀 Performance Optimization

### AWS Graviton Instances

| Instance Type | vCPUs | Memory | Recommended Use |
|---------------|-------|--------|-----------------|
| `m6g.medium` | 1 | 4 GB | Small models (< 1B params) |
| `m6g.large` | 2 | 8 GB | Medium models (1-3B params) |
| `m6g.xlarge` | 4 | 16 GB | Large models (3-7B params) |
| `m6g.2xlarge` | 8 | 32 GB | Very large models (7B+ params) |

### Memory Requirements

| Model Size | RAM Needed | With Quantization |
|------------|------------|-------------------|
| 125M params | 1 GB | 0.5 GB |
| 350M params | 2 GB | 1 GB |
| 1.5B params | 6 GB | 3 GB |
| 6.7B params | 26 GB | 13 GB |

## 🛠️ Development

### Project Structure

```
llm-arm-inference/
├── app/
│   ├── __init__.py
│   ├── model_loader.py     # Platform detection & model loading
│   ├── inference.py        # Inference engine
│   └── server.py           # FastAPI application
├── kubernetes/
│   └── deployment-arm.yaml # Kubernetes deployment
├── scripts/
│   └── build_push.sh       # Multi-arch Docker build
├── Dockerfile              # Multi-stage container
├── requirements.txt        # Python dependencies
├── requirements-fallback.txt # Alternative dependencies
├── setup.py               # Installation script
├── .dockerignore          # Docker ignore rules
└── README.md              # This file
```

### Testing

```bash
# Test imports
python -c "import app.model_loader; import app.inference; import app.server; print('✓ All modules imported successfully')"

# Test model loading
python -c "from app.model_loader import ModelLoader; ml = ModelLoader(); print('✓ Model loader initialized')"

# Test API endpoints
python app/server.py &
sleep 5
curl http://localhost:8000/health
```

### Adding New Models

1. Update the model name in environment variables
2. Ensure the model is compatible with Transformers
3. Test with different prompt formats
4. Adjust memory limits if needed

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   
   # Use fallback requirements if needed
   pip install -r requirements-fallback.txt
   ```

2. **Memory Issues:**
   ```bash
   # Use a smaller model
   export MODEL_NAME="microsoft/DialoGPT-small"
   
   # Enable quantization (ARM only)
   # Automatic based on platform detection
   ```

3. **Model Loading Fails:**
   ```bash
   # Check HuggingFace token
   export HF_TOKEN="your_token_here"
   
   # Verify model name
   curl -s "https://huggingface.co/api/models/microsoft/DialoGPT-medium"
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python app/server.py
```

### Windows-Specific Notes

- Use PowerShell or Command Prompt
- Ensure Python 3.11+ is installed
- Virtual environment recommended
- Some packages may require Visual Studio Build Tools

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and community support
- **Documentation**: Check the `/docs` directory for detailed guides

---

**🏗️ Built for modern cloud infrastructure - ARM and x86_64 ready!**
