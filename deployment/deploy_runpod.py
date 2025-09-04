#!/usr/bin/env python3
"""RunPod deployment script for Mamba-KAN pipeline."""

import os
import sys
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    raise ValueError("RUNPOD_API_KEY not found in .env file")

RUNPOD_API_BASE = "https://api.runpod.ai/graphql"

class RunPodDeployer:
    """RunPod deployment manager for Mamba-KAN project."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def graphql_request(self, query: str, variables: dict = None):
        """Execute GraphQL request to RunPod API."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
            
        response = requests.post(
            RUNPOD_API_BASE,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"GraphQL request failed: {response.status_code}, {response.text}")
            
        return response.json()
    
    def get_gpu_types(self):
        """Get available GPU types."""
        query = """
        query {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                lowestPrice {
                    minimumBidPrice
                    uninterruptablePrice
                }
            }
        }
        """
        
        result = self.graphql_request(query)
        return result["data"]["gpuTypes"]
    
    def create_pod(self, gpu_type_id: str, image: str = "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"):
        """Create a new RunPod instance."""
        
        startup_script = """#!/bin/bash
# RunPod startup script for Mamba-KAN
cd /workspace
git clone https://github.com/stchakwdev/Mamba_KAN.git
cd Mamba_KAN
bash deployment/setup_environment.sh
python deployment/run_tests.py
"""

        query = """
        mutation podFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput) {
            podFindAndDeployOnDemand(input: $input) {
                id
                imageName
                env
                machineId
                machine {
                    podHostId
                }
            }
        }
        """
        
        variables = {
            "input": {
                "cloudType": "SECURE",
                "gpuTypeId": gpu_type_id,
                "name": "mamba-kan-test",
                "imageName": image,
                "dockerArgs": "",
                "ports": "8888/http,22/tcp",
                "volumeInGb": 30,
                "containerDiskInGb": 20,
                "env": [
                    {"key": "JUPYTER_PASSWORD", "value": "mamba_kan_2024"},
                    {"key": "RUNPOD_POD_ID", "value": "{{RUNPOD_POD_ID}}"}
                ],
                "startupScript": startup_script
            }
        }
        
        result = self.graphql_request(query, variables)
        return result["data"]["podFindAndDeployOnDemand"]
    
    def get_pod_status(self, pod_id: str):
        """Get status of a RunPod instance."""
        query = """
        query pod($input: PodIdInput) {
            pod(input: $input) {
                id
                name
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
                machine {
                    podHostId
                }
            }
        }
        """
        
        variables = {"input": {"podId": pod_id}}
        result = self.graphql_request(query, variables)
        return result["data"]["pod"]
    
    def stop_pod(self, pod_id: str):
        """Stop a RunPod instance."""
        query = """
        mutation podStop($input: PodStopInput) {
            podStop(input: $input) {
                id
                desiredStatus
            }
        }
        """
        
        variables = {"input": {"podId": pod_id}}
        result = self.graphql_request(query, variables)
        return result["data"]["podStop"]
    
    def deploy_and_test(self):
        """Full deployment and testing pipeline."""
        print("🚀 Starting Mamba-KAN RunPod Deployment")
        print("=" * 60)
        
        # Get available GPUs
        print("Getting available GPU types...")
        gpu_types = self.get_gpu_types()
        
        # Find RTX 3090 or RTX 4090 (good price/performance for research)
        suitable_gpus = [
            gpu for gpu in gpu_types 
            if any(name in gpu["displayName"].lower() for name in ["rtx 3090", "rtx 4090", "a100"])
            and gpu["secureCloud"]
        ]
        
        if not suitable_gpus:
            print("❌ No suitable GPUs available")
            return None
            
        # Sort by price and pick cheapest
        suitable_gpus.sort(key=lambda x: float(x["lowestPrice"]["uninterruptablePrice"]))
        chosen_gpu = suitable_gpus[0]
        
        print(f"🎯 Selected GPU: {chosen_gpu['displayName']}")
        print(f"💰 Price: ${chosen_gpu['lowestPrice']['uninterruptablePrice']}/hr")
        print(f"💾 Memory: {chosen_gpu['memoryInGb']} GB")
        
        # Create pod
        print("\n⚡ Creating RunPod instance...")
        pod_result = self.create_pod(chosen_gpu["id"])
        
        if not pod_result:
            print("❌ Failed to create pod")
            return None
            
        pod_id = pod_result["id"]
        print(f"✅ Pod created: {pod_id}")
        
        # Wait for pod to be ready
        print("\n⏳ Waiting for pod to be ready...")
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            pod_status = self.get_pod_status(pod_id)
            
            if pod_status and pod_status["runtime"]:
                runtime = pod_status["runtime"]
                
                # Check if Jupyter port is available
                jupyter_port = None
                for port in runtime.get("ports", []):
                    if port["privatePort"] == 8888:
                        jupyter_port = port
                        break
                
                if jupyter_port:
                    jupyter_url = f"https://{jupyter_port['ip']}:{jupyter_port['publicPort']}"
                    print(f"🎉 Pod is ready!")
                    print(f"🔗 Jupyter URL: {jupyter_url}")
                    print(f"🔑 Password: mamba_kan_2024")
                    
                    return {
                        "pod_id": pod_id,
                        "jupyter_url": jupyter_url,
                        "ssh_info": runtime.get("ports", [])
                    }
            
            print(".", end="", flush=True)
            time.sleep(10)
        
        print("\n❌ Pod did not become ready within timeout")
        self.stop_pod(pod_id)
        return None


def main():
    """Main deployment function."""
    deployer = RunPodDeployer(RUNPOD_API_KEY)
    
    try:
        deployment_info = deployer.deploy_and_test()
        
        if deployment_info:
            pod_id = deployment_info["pod_id"]
            
            print("\n" + "=" * 80)
            print("🎯 DEPLOYMENT SUCCESSFUL")
            print("=" * 80)
            print(f"Pod ID: {pod_id}")
            print(f"Jupyter URL: {deployment_info['jupyter_url']}")
            
            print("\n📝 Next Steps:")
            print("1. Open Jupyter in browser")
            print("2. Navigate to /workspace/Mamba_KAN")
            print("3. Run: python deployment/run_tests.py")
            print("4. Check test results and benchmarks")
            
            input("\nPress Enter when done to stop the pod...")
            deployer.stop_pod(pod_id)
            print("🛑 Pod stopped")
        
    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted by user")
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()