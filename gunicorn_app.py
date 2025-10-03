"""Standalone Gunicorn application runner for Threadboard."""

import os
import sys
import multiprocessing
import gunicorn.app.base

from app import app as application


def get_optimal_config():
    """Get optimal Gunicorn configuration based on detected CPU cores and environment."""
    cpu_count = multiprocessing.cpu_count()
    env = os.getenv("ENV", "prod")

    print(f"Detected {cpu_count} CPU cores, environment: {env}")

    # For e2-micro (1 vCPU, 1GB RAM): Conservative settings
    if cpu_count == 1:
        workers = 2  # Minimum for availability
        threads = 4  # Total 8 concurrent requests
        max_requests = 200
        max_requests_jitter = 50
    elif cpu_count == 2:
        workers = 3
        threads = 4  # Total 12 concurrent requests
        max_requests = 350
        max_requests_jitter = 75
    else:
        # 4+ CPUs: More aggressive scaling
        workers = min((2 * cpu_count) + 1, 9)
        threads = 4
        max_requests = 500
        max_requests_jitter = 100

    config = {
        "workers": workers,
        "threads": threads,
        "max_requests": max_requests,
        "max_requests_jitter": max_requests_jitter,
        "preload_app": env == "prod",
    }

    total_concurrent = workers * threads
    print(
        f"Configuration: {workers} workers × {threads} threads = "
        f"{total_concurrent} concurrent requests"
    )

    return config


class StandaloneApplication(gunicorn.app.base.BaseApplication):
    """Standalone Gunicorn application wrapper."""

    def __init__(self, app, app_options=None):
        self.options = app_options or {}
        self.application = app
        super().__init__()

    def init(self, parser, opts, args):
        """Initialize the application with command line arguments."""
        return None

    def load_config(self):
        """Load configuration from options."""
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        """Load the WSGI application."""
        return self.application


class HealthCheckFilter:
    """Filter to suppress health check requests from access logs."""

    def write(self, data):
        """Write data to stdout, filtering out health check requests."""
        if '"GET /health' not in data:
            sys.stdout.write(data)
            sys.stdout.flush()

    def flush(self):
        """Flush method required by logger interface."""
        pass


if __name__ == "__main__":
    optimal_config = get_optimal_config()
    port = int(os.getenv("PORT", 5000))
    ssl_port = int(os.getenv("SSL_PORT", 5443))
    ssl_cert = os.getenv("SSL_CERT")
    ssl_key = os.getenv("SSL_KEY")

    # Base configuration
    gunicorn_options = {
        "worker_class": "gthread",  # Enable multi-threading for I/O-bound app
        "timeout": 30,
        "graceful_timeout": 55,
        "keepalive": 5,
        "access_log": "-",
        "access_logfile": "-",
        "access_logger": HealthCheckFilter(),
        "error_log": "-",
        "log_level": "info",
        **optimal_config,
    }

    # Check if SSL is configured
    if ssl_cert and ssl_key and os.path.exists(ssl_cert) and os.path.exists(ssl_key):
        # Bind to both HTTP and HTTPS
        gunicorn_options["bind"] = [
            f"0.0.0.0:{port}",
            f"0.0.0.0:{ssl_port}"
        ]
        gunicorn_options["certfile"] = ssl_cert
        gunicorn_options["keyfile"] = ssl_key
        print(f"Starting Gunicorn with SSL on ports {port} (HTTP) and {ssl_port} (HTTPS)")
    else:
        # HTTP only
        gunicorn_options["bind"] = f"0.0.0.0:{port}"
        print(f"Starting Gunicorn on port {port} (HTTP only)")

    print(f"Configuration: {optimal_config}")
    StandaloneApplication(application, gunicorn_options).run()
