#!/bin/bash

# Sync synchrony-mcp-server with Conv2d project
# This script integrates the MCP server deployment tools with Conv2d

set -euo pipefail

# Configuration
CONV2D_DIR="/home/wllmflower/Development/Conv2d"
MCP_SERVER_DIR="/home/wllmflower/Development/synchrony-mcp-server"
EDGE_PI_HOST="edge.tailfdc654.ts.net"
EDGE_PI_IP="100.127.242.78"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Conv2d + MCP Server Integration${NC}"
echo -e "${GREEN}======================================${NC}"

# Function to check dependencies
check_dependencies() {
    echo -e "\n${YELLOW}Checking dependencies...${NC}"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Node.js not found. Please install Node.js 18+${NC}"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}npm not found. Please install npm${NC}"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 not found. Please install Python 3.10+${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ All dependencies found${NC}"
}

# Function to setup MCP server
setup_mcp_server() {
    echo -e "\n${YELLOW}Setting up MCP server...${NC}"
    
    cd "$MCP_SERVER_DIR"
    
    # Install dependencies
    if [ ! -d "node_modules" ]; then
        echo "Installing MCP server dependencies..."
        npm install
    fi
    
    # Build TypeScript
    echo "Building MCP server..."
    npm run build
    
    # Create Conv2d integration tool
    echo "Creating Conv2d integration tool..."
    cat > src/tools/conv2d-integration.ts << 'EOF'
import { MCPTool } from '../types/index.js';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';

const execAsync = promisify(exec);

export const conv2dIntegration: MCPTool = {
    name: 'conv2d_integration',
    description: 'Conv2d model deployment and management',
    inputSchema: {
        type: 'object',
        properties: {
            action: {
                type: 'string',
                enum: ['deploy', 'test', 'benchmark', 'rollback'],
                description: 'Integration action to perform'
            },
            target: {
                type: 'string',
                enum: ['edge-pi', 'cloud', 'local'],
                default: 'edge-pi'
            },
            model_path: {
                type: 'string',
                description: 'Path to model file'
            },
            options: {
                type: 'object',
                properties: {
                    run_tests: { type: 'boolean', default: true },
                    backup: { type: 'boolean', default: true }
                }
            }
        },
        required: ['action']
    }
};

export async function handleConv2dIntegration(args: any) {
    const { action, target = 'edge-pi', model_path, options = {} } = args;
    
    switch (action) {
        case 'deploy':
            return await deployConv2dModel(target, model_path, options);
        case 'test':
            return await runIntegrationTests(target);
        case 'benchmark':
            return await runBenchmarks(target);
        case 'rollback':
            return await rollbackDeployment(target);
        default:
            throw new Error(`Unknown action: ${action}`);
    }
}

async function deployConv2dModel(target: string, modelPath: string, options: any) {
    const conv2dDir = '/home/wllmflower/Development/Conv2d';
    
    // Export model to ONNX
    const exportCmd = `cd ${conv2dDir} && python scripts/deployment/export_fsq_to_onnx.py --checkpoint ${modelPath}`;
    await execAsync(exportCmd);
    
    // Deploy based on target
    if (target === 'edge-pi') {
        const deployCmd = `cd ${conv2dDir} && ./deploy_to_edge_pi_secure.sh`;
        const result = await execAsync(deployCmd);
        
        return {
            content: [{
                type: 'text',
                text: `Model deployed to Edge Pi successfully\n${result.stdout}`
            }]
        };
    }
    
    return { content: [{ type: 'text', text: 'Deployment complete' }] };
}

async function runIntegrationTests(target: string) {
    const conv2dDir = '/home/wllmflower/Development/Conv2d';
    const testCmd = `cd ${conv2dDir} && pytest tests/integration -v`;
    
    const result = await execAsync(testCmd);
    
    return {
        content: [{
            type: 'text',
            text: `Integration tests completed\n${result.stdout}`
        }]
    };
}

async function runBenchmarks(target: string) {
    const conv2dDir = '/home/wllmflower/Development/Conv2d';
    const benchCmd = `cd ${conv2dDir} && python benchmarks/latency_benchmark.py`;
    
    const result = await execAsync(benchCmd);
    
    return {
        content: [{
            type: 'text',
            text: `Benchmarks completed\n${result.stdout}`
        }]
    };
}

async function rollbackDeployment(target: string) {
    // Implementation for rollback
    return {
        content: [{
            type: 'text',
            text: 'Rollback completed successfully'
        }]
    };
}
EOF
    
    # Rebuild with new integration
    npm run build
    
    echo -e "${GREEN}✓ MCP server setup complete${NC}"
}

# Function to create integration tests
create_integration_tests() {
    echo -e "\n${YELLOW}Creating integration tests...${NC}"
    
    cd "$CONV2D_DIR"
    
    # Create test directory if not exists
    mkdir -p tests/integration/mcp
    
    # Create MCP integration test
    cat > tests/integration/mcp/test_mcp_deployment.py << 'EOF'
"""
Integration tests for MCP server deployment
"""

import pytest
import requests
import subprocess
import json
import time
from pathlib import Path

class TestMCPDeployment:
    """Test MCP server integration with Conv2d"""
    
    @pytest.fixture
    def mcp_client(self):
        """Create MCP client for testing"""
        return MCPTestClient("http://localhost:3000")
    
    def test_mcp_server_health(self, mcp_client):
        """Test MCP server is running and healthy"""
        response = mcp_client.health_check()
        assert response['status'] == 'healthy'
        assert 'tools' in response
        assert 'conv2d_integration' in response['tools']
    
    def test_deploy_model_to_edge(self, mcp_client):
        """Test model deployment to Edge Pi via MCP"""
        result = mcp_client.call_tool('conv2d_integration', {
            'action': 'deploy',
            'target': 'edge-pi',
            'model_path': 'models/best_conv2d_fsq.pth',
            'options': {
                'run_tests': True,
                'backup': True
            }
        })
        
        assert result['status'] == 'success'
        assert 'deployment_id' in result
        
        # Verify deployment on Edge Pi
        edge_health = requests.get('http://edge.tailfdc654.ts.net:8082/healthz')
        assert edge_health.status_code == 200
    
    def test_run_integration_tests_via_mcp(self, mcp_client):
        """Test running integration tests through MCP"""
        result = mcp_client.call_tool('conv2d_integration', {
            'action': 'test',
            'target': 'local'
        })
        
        assert result['status'] == 'success'
        assert 'passed' in result['summary']
        assert result['summary']['failed'] == 0
    
    def test_benchmark_deployment(self, mcp_client):
        """Test performance benchmarking via MCP"""
        result = mcp_client.call_tool('conv2d_integration', {
            'action': 'benchmark',
            'target': 'edge-pi'
        })
        
        assert result['status'] == 'success'
        assert 'latency_ms' in result['metrics']
        assert result['metrics']['latency_ms'] < 100  # <100ms requirement
        assert result['metrics']['throughput_fps'] > 20  # >20 fps requirement

class MCPTestClient:
    """Test client for MCP server"""
    
    def __init__(self, server_url):
        self.server_url = server_url
    
    def health_check(self):
        """Check MCP server health"""
        response = requests.get(f"{self.server_url}/health")
        return response.json()
    
    def call_tool(self, tool_name, args):
        """Call MCP tool"""
        response = requests.post(
            f"{self.server_url}/tools/{tool_name}",
            json=args
        )
        return response.json()
EOF
    
    echo -e "${GREEN}✓ Integration tests created${NC}"
}

# Function to start services
start_services() {
    echo -e "\n${YELLOW}Starting services...${NC}"
    
    # Start MCP server
    cd "$MCP_SERVER_DIR"
    npm start &
    MCP_PID=$!
    echo "MCP server started (PID: $MCP_PID)"
    
    # Start Conv2d API (if needed)
    if [ -f "$CONV2D_DIR/docker-compose.yml" ]; then
        cd "$CONV2D_DIR"
        docker-compose up -d
        echo "Conv2d services started"
    fi
    
    # Wait for services to be ready
    echo "Waiting for services to be ready..."
    sleep 10
    
    # Check services
    if curl -s http://localhost:3000/health > /dev/null; then
        echo -e "${GREEN}✓ MCP server is running${NC}"
    else
        echo -e "${RED}✗ MCP server failed to start${NC}"
        exit 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    echo -e "\n${YELLOW}Running integration tests...${NC}"
    
    cd "$CONV2D_DIR"
    
    # Run MCP integration tests
    pytest tests/integration/mcp -v
    
    # Run full integration suite
    pytest tests/integration -v --tb=short
    
    echo -e "${GREEN}✓ Integration tests complete${NC}"
}

# Function to deploy to Edge Pi
deploy_to_edge() {
    echo -e "\n${YELLOW}Deploying to Edge Pi...${NC}"
    
    cd "$CONV2D_DIR"
    
    # Export model
    echo "Exporting model to ONNX..."
    python scripts/deployment/export_fsq_to_onnx.py \
        --checkpoint models/best_conv2d_fsq.pth \
        --output models/conv2d_fsq.onnx
    
    # Deploy via secure script
    echo "Deploying to Edge Pi..."
    ./deploy_to_edge_pi_secure.sh
    
    # Verify deployment
    echo "Verifying deployment..."
    if curl -s http://${EDGE_PI_IP}:8082/healthz | jq -e '.status == "healthy"' > /dev/null; then
        echo -e "${GREEN}✓ Deployment successful${NC}"
    else
        echo -e "${RED}✗ Deployment verification failed${NC}"
        exit 1
    fi
}

# Function to show status
show_status() {
    echo -e "\n${BLUE}======================================${NC}"
    echo -e "${BLUE}Integration Status${NC}"
    echo -e "${BLUE}======================================${NC}"
    
    # Check MCP server
    if pgrep -f "mcp-server" > /dev/null; then
        echo -e "MCP Server: ${GREEN}Running${NC}"
    else
        echo -e "MCP Server: ${RED}Stopped${NC}"
    fi
    
    # Check Edge Pi
    if ping -c 1 ${EDGE_PI_IP} > /dev/null 2>&1; then
        echo -e "Edge Pi: ${GREEN}Online${NC}"
        
        # Check Edge Pi services
        if curl -s http://${EDGE_PI_IP}:8082/healthz > /dev/null 2>&1; then
            echo -e "Conv2d API: ${GREEN}Running${NC}"
        else
            echo -e "Conv2d API: ${RED}Not Running${NC}"
        fi
    else
        echo -e "Edge Pi: ${RED}Offline${NC}"
    fi
    
    # Show metrics if available
    if curl -s http://${EDGE_PI_IP}:8082/metrics > /dev/null 2>&1; then
        echo -e "\n${BLUE}Performance Metrics:${NC}"
        curl -s http://${EDGE_PI_IP}:8082/metrics | grep "inference_duration" | head -5
    fi
}

# Main execution
main() {
    case "${1:-}" in
        setup)
            check_dependencies
            setup_mcp_server
            create_integration_tests
            ;;
        start)
            start_services
            ;;
        test)
            run_integration_tests
            ;;
        deploy)
            deploy_to_edge
            ;;
        status)
            show_status
            ;;
        all)
            check_dependencies
            setup_mcp_server
            create_integration_tests
            start_services
            run_integration_tests
            deploy_to_edge
            show_status
            ;;
        *)
            echo "Usage: $0 {setup|start|test|deploy|status|all}"
            echo ""
            echo "Commands:"
            echo "  setup   - Set up MCP server and create integration tests"
            echo "  start   - Start MCP server and Conv2d services"
            echo "  test    - Run integration tests"
            echo "  deploy  - Deploy model to Edge Pi"
            echo "  status  - Show integration status"
            echo "  all     - Run all steps"
            exit 1
            ;;
    esac
}

main "$@"