variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-west1"
}

variable "my_ip_address" {
  description = "Your IP address for SSH access (CIDR format, e.g., '1.2.3.4/32')"
  type        = string
}

variable "reddit_client_id" {
  description = "Reddit OAuth Client ID"
  type        = string
  sensitive   = true
}

variable "reddit_client_secret" {
  description = "Reddit OAuth Client Secret"
  type        = string
  sensitive   = true
}

variable "gemini_api_key" {
  description = "Google Gemini API Key"
  type        = string
  sensitive   = true
}

variable "cloudflare_api_key" {
  description = "Cloudflare API Key (optional - required for DNS configuration)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "cloudflare_zone_id" {
  description = "Cloudflare Zone name/domain (e.g., 'example.com') - leave empty to skip DNS configuration"
  type        = string
  default     = ""
}

variable "cloudflare_record_name" {
  description = "DNS record name (e.g., 'threadboard' or '@' for root domain)"
  type        = string
  default     = "@"
}
