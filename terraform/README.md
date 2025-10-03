# Threadboard GCP Infrastructure

This directory contains Terraform configuration to deploy Threadboard on Google Cloud Platform.

## Prerequisites

1. **GCP Project**: Create a GCP project
2. **gcloud CLI**: Install and authenticate with `gcloud auth application-default login`
3. **Terraform**: Install Terraform >= 1.0
4. **Reddit API Credentials**: Create a Reddit app at https://www.reddit.com/prefs/apps
5. **Gemini API Key**: Get from https://aistudio.google.com/app/apikey

## Setup

1. **Copy the example tfvars file**:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. **Edit `terraform.tfvars`** with your values:
   - `project_id`: Your GCP project ID
   - `region`: GCP region (default: us-west1)
   - `my_ip_address`: Your IP in CIDR format (find with `curl ifconfig.me`)
   - `reddit_client_id` and `reddit_client_secret`: From Reddit app
   - `gemini_api_key`: From Google AI Studio

3. **Initialize Terraform**:
   ```bash
   terraform init
   ```

4. **Review the plan**:
   ```bash
   terraform plan
   ```

5. **Apply the configuration**:
   ```bash
   terraform apply
   ```

## What Gets Created

- **Compute Instance**: e2-micro VM running Container-Optimized OS (free tier eligible)
- **Static IP**: External IP for the application
- **Service Account**: With permissions to access Secret Manager
- **Firewall Rules**: Allow HTTP (5000), HTTPS, and SSH
- **Artifact Registry**: Docker repository for container images
- **Secret Manager Secrets**: Secure storage for API keys and credentials

## Deployment

After Terraform creates the infrastructure:

1. **Connect your GitHub repository** to Cloud Build in GCP Console
2. **Create a Cloud Build trigger** that runs on push to main branch
3. The `cloudbuild.yaml` in the root directory will:
   - Build the Docker image
   - Push to Artifact Registry
   - Signal the VM to update automatically

## Accessing the Application

After deployment, get the external IP:

```bash
terraform output external_ip
```

Visit: `http://<EXTERNAL_IP>:5000`

## Management

SSH into the instance:
```bash
gcloud compute ssh threadboard-instance --zone=us-west1-a
```

View logs:
```bash
docker logs threadboard
```

Manual update:
```bash
sudo /var/lib/toolbox/update-app
```

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

## Costs

- e2-micro instance: ~$7/month (or free with free tier)
- Static IP: Free while in use
- Artifact Registry: First 0.5GB free, then ~$0.10/GB/month
- Secret Manager: First 10,000 operations free monthly

Total estimated cost: ~$7/month or free with GCP free tier.
