output "instance_name" {
  description = "Name of the Compute Engine instance"
  value       = google_compute_instance.threadboard_instance.name
}

output "instance_zone" {
  description = "Zone of the Compute Engine instance"
  value       = google_compute_instance.threadboard_instance.zone
}

output "external_ip" {
  description = "External IP address of the instance"
  value       = google_compute_address.threadboard_static_ip.address
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.threadboard_service_account.email
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.threadboard.repository_id}"
}

output "app_url" {
  description = "Application URL"
  value       = var.cloudflare_zone_id != "" ? "https://${cloudflare_record.threadboard[0].hostname}" : "http://${google_compute_address.threadboard_static_ip.address}"
}

output "cloudflare_dns_record" {
  description = "Cloudflare DNS record (if configured)"
  value       = var.cloudflare_zone_id != "" ? cloudflare_record.threadboard[0].hostname : "Not configured"
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "gcloud compute ssh ${google_compute_instance.threadboard_instance.name} --project=${var.project_id} --zone=${google_compute_instance.threadboard_instance.zone}"
}
