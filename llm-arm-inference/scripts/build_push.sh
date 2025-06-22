#!/bin/bash

# Multi-architecture Docker build and push script for LLM ARM Inference API
# This script builds and pushes Docker images for both AMD64 and ARM64 architectures

set -e

# Configuration
IMAGE_NAME=${IMAGE_NAME:-"llm-arm-inference"}
REGISTRY=${REGISTRY:-"your-registry.com"}
TAG=${TAG:-"latest"}
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    echo_error "Docker buildx is not available. Please install Docker Desktop or enable buildx."
    exit 1
fi

# Create a new builder instance for multi-arch builds
echo_info "Setting up Docker buildx for multi-architecture builds..."
docker buildx create --name llm-builder --use --bootstrap || true

# Verify buildx supports required platforms
echo_info "Checking supported platforms..."
docker buildx inspect --bootstrap

echo_info "Building multi-architecture image: ${FULL_IMAGE_NAME}"
echo_info "Platforms: linux/amd64,linux/arm64"

# Build and push multi-architecture image
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag "${FULL_IMAGE_NAME}" \
    --tag "${REGISTRY}/${IMAGE_NAME}:latest" \
    --push \
    --progress=plain \
    .

if [ $? -eq 0 ]; then
    echo_success "Successfully built and pushed multi-architecture image!"
    echo_info "Image: ${FULL_IMAGE_NAME}"
    echo_info "Platforms: linux/amd64, linux/arm64"
    
    # Verify the manifest
    echo_info "Verifying multi-architecture manifest..."
    docker buildx imagetools inspect "${FULL_IMAGE_NAME}"
else
    echo_error "Build failed!"
    exit 1
fi

# Optional: Build platform-specific images with different tags
if [ "${BUILD_PLATFORM_SPECIFIC}" = "true" ]; then
    echo_info "Building platform-specific images..."
    
    # AMD64 image
    echo_info "Building AMD64 image..."
    docker buildx build \
        --platform linux/amd64 \
        --tag "${REGISTRY}/${IMAGE_NAME}:${TAG}-amd64" \
        --push \
        .
    
    # ARM64 image
    echo_info "Building ARM64 image..."
    docker buildx build \
        --platform linux/arm64 \
        --tag "${REGISTRY}/${IMAGE_NAME}:${TAG}-arm64" \
        --push \
        .
    
    echo_success "Platform-specific images built successfully!"
fi

# Test the image locally (AMD64 only for development)
if [ "${TEST_LOCAL}" = "true" ]; then
    echo_info "Testing image locally..."
    docker run --rm -d \
        --name llm-test \
        -p 8000:8000 \
        "${FULL_IMAGE_NAME}"
    
    # Wait for service to start
    echo_info "Waiting for service to start..."
    sleep 30
    
    # Test health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo_success "Health check passed!"
    else
        echo_warning "Health check failed - service might still be starting"
    fi
    
    # Stop test container
    docker stop llm-test
fi

echo_success "Build and push completed successfully!"
echo_info "To deploy on ARM nodes, use: kubectl apply -f kubernetes/deployment-arm.yaml"
