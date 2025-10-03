terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
    http = {
      source  = "hashicorp/http"
      version = "~> 3.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "cloudflare" {
  api_token = var.cloudflare_api_key
}

# Enable required APIs
resource "google_project_service" "enable_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage-component.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# Data sources
data "google_project" "project" {
  project_id = var.project_id
}

# Fetch current Cloudflare IP ranges
data "http" "cloudflare_ipv4" {
  url = "https://www.cloudflare.com/ips-v4"
}

data "http" "cloudflare_ipv6" {
  url = "https://www.cloudflare.com/ips-v6"
}

# Parse Cloudflare IP ranges
locals {
  cloudflare_ipv4_ranges = split("\n", trimspace(data.http.cloudflare_ipv4.response_body))
  cloudflare_ipv6_ranges = split("\n", trimspace(data.http.cloudflare_ipv6.response_body))
  cloudflare_ip_ranges   = concat(local.cloudflare_ipv4_ranges, local.cloudflare_ipv6_ranges)
}

# Static IP for threadboard instance
resource "google_compute_address" "threadboard_static_ip" {
  depends_on = [google_project_service.enable_apis]

  name         = "threadboard-ip"
  region       = var.region
  network_tier = "STANDARD"
}

# Service account for threadboard instance
resource "google_service_account" "threadboard_service_account" {
  depends_on = [google_project_service.enable_apis]

  account_id   = "threadboard"
  display_name = "Threadboard Service Account"
  description  = "Service account for Threadboard application"
}

# IAM roles for threadboard service account
resource "google_project_iam_member" "threadboard_secret_accessor" {
  depends_on = [google_project_service.enable_apis]

  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.threadboard_service_account.email}"
}

resource "google_project_iam_member" "threadboard_logging_writer" {
  depends_on = [google_project_service.enable_apis]

  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.threadboard_service_account.email}"
}

resource "google_project_iam_member" "threadboard_monitoring_writer" {
  depends_on = [google_project_service.enable_apis]

  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.threadboard_service_account.email}"
}

resource "google_project_iam_member" "threadboard_artifact_reader" {
  depends_on = [google_project_service.enable_apis]

  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.threadboard_service_account.email}"
}

# Reddit API credentials
resource "google_secret_manager_secret" "reddit_client_id" {
  depends_on = [google_project_service.enable_apis]

  secret_id = "reddit-client-id"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "reddit_client_id" {
  secret      = google_secret_manager_secret.reddit_client_id.id
  secret_data = var.reddit_client_id
}

resource "google_secret_manager_secret" "reddit_client_secret" {
  depends_on = [google_project_service.enable_apis]

  secret_id = "reddit-client-secret"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "reddit_client_secret" {
  secret      = google_secret_manager_secret.reddit_client_secret.id
  secret_data = var.reddit_client_secret
}

# Gemini API Key
resource "google_secret_manager_secret" "gemini_api_key" {
  depends_on = [google_project_service.enable_apis]

  secret_id = "gemini-api-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "gemini_api_key" {
  secret      = google_secret_manager_secret.gemini_api_key.id
  secret_data = var.gemini_api_key
}

# Flask secret key
resource "random_password" "flask_secret_key" {
  length  = 32
  special = true
}

resource "google_secret_manager_secret" "flask_secret_key" {
  depends_on = [google_project_service.enable_apis]

  secret_id = "flask-secret-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "flask_secret_key" {
  secret      = google_secret_manager_secret.flask_secret_key.id
  secret_data = random_password.flask_secret_key.result
}

# Firewall rule for HTTP/HTTPS traffic - IPv4 (Cloudflare only)
resource "google_compute_firewall" "threadboard_http_ipv4" {
  depends_on = [google_project_service.enable_apis]

  name    = "threadboard-allow-http-cloudflare-ipv4"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = local.cloudflare_ipv4_ranges
  target_tags   = ["threadboard"]

  description = "Allow HTTP/HTTPS from Cloudflare IPv4 ranges only"
}

# Firewall rule for HTTP/HTTPS traffic - IPv6 (Cloudflare only)
resource "google_compute_firewall" "threadboard_http_ipv6" {
  depends_on = [google_project_service.enable_apis]

  name    = "threadboard-allow-http-cloudflare-ipv6"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = local.cloudflare_ipv6_ranges
  target_tags   = ["threadboard"]

  description = "Allow HTTP/HTTPS from Cloudflare IPv6 ranges only"
}

resource "google_compute_firewall" "threadboard_ssh" {
  depends_on = [google_project_service.enable_apis]

  name    = "threadboard-allow-ssh"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = [var.my_ip_address]
  target_tags   = ["threadboard"]
}

# Threadboard Compute Engine instance
resource "google_compute_instance" "threadboard_instance" {
  depends_on = [google_project_service.enable_apis, google_compute_address.threadboard_static_ip]

  name         = "threadboard-instance"
  machine_type = "e2-micro" # Free tier eligible
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
      size  = 10
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip       = google_compute_address.threadboard_static_ip.address
      network_tier = "STANDARD"
    }
  }

  service_account {
    email  = google_service_account.threadboard_service_account.email
    scopes = ["cloud-platform"]
  }

  tags = ["threadboard"]

  metadata = {
    environment    = "production"
    enable-oslogin = "TRUE"
  }

  # Startup script for threadboard instance
  metadata_startup_script = templatefile("${path.module}/startup-script.sh", {
    region                  = var.region
    project_id              = var.project_id
    project_number          = data.google_project.project.number
    reddit_client_id_secret = "reddit-client-id"
    reddit_client_secret_secret = "reddit-client-secret"
    gemini_api_key_secret   = "gemini-api-key"
    flask_secret_key_secret = "flask-secret-key"
  })
}

# IAM permissions for Cloud Build to deploy to threadboard instance
resource "google_project_iam_member" "cloudbuild_compute_ssh" {
  depends_on = [google_project_service.enable_apis]

  project = var.project_id
  role    = "roles/compute.osLogin"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_compute_viewer" {
  depends_on = [google_project_service.enable_apis]

  project = var.project_id
  role    = "roles/compute.viewer"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_service_account_user" {
  depends_on = [google_project_service.enable_apis]

  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "threadboard" {
  depends_on = [google_project_service.enable_apis]

  location      = var.region
  repository_id = "threadboard"
  format        = "DOCKER"
}

# IAM permissions for Cloud Build to access Artifact Registry
resource "google_project_iam_member" "cloudbuild_artifacts_writer" {
  depends_on = [google_project_service.enable_apis]

  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

# Cloudflare DNS record (optional - only created if cloudflare_zone_id is provided)
resource "cloudflare_record" "threadboard" {
  count = var.cloudflare_zone_id != "" ? 1 : 0

  zone_id = var.cloudflare_zone_id
  name    = var.cloudflare_record_name
  value   = google_compute_address.threadboard_static_ip.address
  type    = "A"
  proxied = true
  ttl     = 1 # Auto TTL when proxied
}
