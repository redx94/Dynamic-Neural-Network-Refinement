# Prometheus Configuration

The `prometheus/` directory contains configuration files for [Prometheus](https://prometheus.io/), an open-source monitoring and alerting toolkit. Prometheus is used in the **Dynamic Neural Network Refinement** project to collect, store, and query metrics from various services, enabling effective monitoring and performance analysis.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Configuration Files](#configuration-files)
  - [prometheus.yml](#prometheusyml)
  - [alert.rules.yml](#alertrulesyml)
- [Setting Up Prometheus](#setting-up-prometheus)
- [Integrating with Grafana](#integrating-with-grafana)
- [Alerting](#alerting)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

Prometheus is a powerful tool for monitoring applications, collecting metrics, and setting up alerts based on predefined rules. In this project, Prometheus monitors the FastAPI server, collects custom metrics, and works in tandem with Grafana to provide visual dashboards for real-time insights.

## Directory Structure

```
prometheus/
├── prometheus.yml
└── alert.rules.yml
```

## Configuration Files

### `prometheus.yml`

**Purpose:**  
Defines the main configuration for the Prometheus server, including global settings, scrape configurations, and alerting rules.

**Key Sections:**

- **Global Configuration:**  
  Sets default parameters like scrape interval and evaluation interval.

- **Scrape Configurations:**  
  Specifies the targets (endpoints) from which Prometheus will scrape metrics. In this project, it includes the FastAPI server and any other relevant services.

- **Alerting Configuration:**  
  Defines alerting rules that Prometheus uses to evaluate when certain conditions are met, triggering alerts accordingly.

**Example Configuration:**

```yaml
# prometheus/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['api:8000']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - 'alertmanager:9093'

rule_files:
  - "alert.rules.yml"
```

### `alert.rules.yml`

**Purpose:**  
Contains alerting rules that define conditions under which alerts should be triggered. These rules help in proactively identifying and responding to issues within the monitored services.

**Key Sections:**

- **Alerts:**  
  Each alert includes a name, severity level, and the condition that triggers the alert.

**Example Configuration:**

```yaml
# prometheus/alert.rules.yml

groups:
  - name: example-alerts
    rules:
      - alert: HighRequestLatency
        expr: histogram_quantile(0.99, sum(rate(app_request_latency_seconds_bucket[5m])) by (le)) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High request latency detected"
          description: "99th percentile latency is above 0.5 seconds for more than 2 minutes."

      - alert: ServiceDown
        expr: up{job="fastapi"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "FastAPI service down"
          description: "The FastAPI service has been down for more than 5 minutes."
```

## Setting Up Prometheus

1. **Navigate to the Prometheus Directory:**

   ```bash
   cd prometheus/
   ```

2. **Run Prometheus Using Docker:**

   ```bash
   docker run -d \
     --name prometheus \
     -p 9090:9090 \
     -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
     -v $(pwd)/alert.rules.yml:/etc/prometheus/alert.rules.yml \
     prom/prometheus:latest
   ```

   **Explanation:**

   - `-d`: Runs the container in detached mode.
   - `--name prometheus`: Names the container `prometheus`.
   - `-p 9090:9090`: Maps port `9090` of the container to port `9090` of the host.
   - `-v`: Mounts the configuration files into the container.
   - `prom/prometheus:latest`: Specifies the Prometheus Docker image.

3. **Verify Prometheus is Running:**

   Access the Prometheus web interface at `http://localhost:9090/`.

## Integrating with Grafana

Grafana is used alongside Prometheus to visualize the collected metrics through customizable dashboards.

1. **Run Grafana Using Docker:**

   ```bash
   docker run -d \
     --name=grafana \
     -p 3000:3000 \
     grafana/grafana:latest
   ```

2. **Access Grafana:**

   Navigate to `http://localhost:3000/` in your web browser.

3. **Add Prometheus as a Data Source:**

   - Log in with the default credentials (`admin` / `admin`).
   - Go to **Configuration** > **Data Sources** > **Add data source**.
   - Select **Prometheus**.
   - Set the URL to `http://prometheus:9090/` or `http://localhost:9090/` depending on your network setup.
   - Click **Save & Test** to verify the connection.

4. **Import Dashboards:**

   - Use pre-built dashboards or create custom ones to visualize metrics like request latency, error rates, and resource utilization.

## Alerting

Prometheus can send alerts to various notification channels through Alertmanager.

1. **Run Alertmanager Using Docker:**

   ```bash
   docker run -d \
     --name alertmanager \
     -p 9093:9093 \
     -v $(pwd)/alertmanager.yml:/etc/alertmanager/alertmanager.yml \
     prom/alertmanager:latest
   ```

2. **Configure Alertmanager:**

   Create an `alertmanager.yml` file to define notification channels (e.g., email, Slack).

   ```yaml
   # prometheus/alertmanager.yml

   global:
     resolve_timeout: 5m

   route:
     receiver: 'slack-notifications'

   receivers:
     - name: 'slack-notifications'
       slack_configs:
         - api_url: 'https://hooks.slack.com/services/your/slack/webhook'
           channel: '#alerts'
           send_resolved: true
   ```

3. **Update Prometheus Configuration:**

   Ensure that the `prometheus.yml` file references Alertmanager correctly.

   ```yaml
   alerting:
     alertmanagers:
       - static_configs:
           - targets:
               - 'alertmanager:9093'
   ```

4. **Reload Prometheus Configuration:**

   Reload the Prometheus configuration to apply changes.

   ```bash
   docker exec -it prometheus kill -HUP 1
   ```

## Best Practices

- **Secure Prometheus and Grafana:**
  - Implement authentication and authorization to restrict access to monitoring dashboards and configurations.
  - Use TLS/SSL to encrypt data in transit.

- **Organize Alerting Rules:**
  - Group related alerts together for better manageability.
  - Clearly define severity levels to prioritize responses.

- **Monitor Prometheus Performance:**
  - Regularly check Prometheus resource usage to prevent performance degradation.
  - Optimize scrape intervals and retention policies based on data needs.

- **Backup Configurations:**
  - Maintain backups of Prometheus and Alertmanager configuration files to facilitate quick recovery in case of failures.

## Troubleshooting

- **Prometheus Not Starting:**
  - Verify that the configuration files (`prometheus.yml` and `alert.rules.yml`) are correctly formatted.
  - Check Docker container logs for error messages:
    
    ```bash
    docker logs prometheus
    ```

- **Metrics Not Appearing in Grafana:**
  - Ensure that Prometheus is running and accessible.
  - Verify that the data source in Grafana is correctly configured.
  - Check for typos in metric names when creating dashboards.

- **Alerts Not Triggering:**
  - Confirm that Alertmanager is running and correctly configured.
  - Validate that alerting rules are correctly defined in `alert.rules.yml`.
  - Check Alertmanager logs for any errors:
    
    ```bash
    docker logs alertmanager
    ```

## Contributing

Contributions to the Prometheus configurations are welcome! To contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/update-prometheus-config
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "chore: update Prometheus scrape configurations"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/update-prometheus-config
   ```

5. **Open a Pull Request**

   Provide a clear description of the changes and their benefits.

For detailed guidelines, refer to the [Best Practices](../docs/best_practices.md) documentation.

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](../LICENSE).

## Contact

For questions, suggestions, or support, please open an issue on the [GitHub repository](https://github.com/redx94/Dynamic-Neural-Network-Refinement/issues) or contact the maintainer at [qtt@null.net](mailto:qtt@null.net).

---
