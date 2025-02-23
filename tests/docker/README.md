# Docker Configuration

The `docker/` directory contains all necessary configurations and scripts to containerize the **Dynamic Neural Network Refinement** project. Containerization ensures consistent environments across different systems, simplifies deployment, and enhances scalability.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Building Docker Images](#building-docker-images)
- [Running Services with Docker Compose](#running-services-with-docker-compose)
- [Managing Containers](#managing-containers)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This directory includes:

- **Dockerfile:** Defines the environment for the FastAPI application, including dependencies and configurations.
- **docker-compose.yml:** Orchestrates multiple Docker containers, including the API server, Prometheus, and Grafana for monitoring.
- **.dockerignore:** Specifies files and directories to exclude from the Docker build context.
- **entrypoint.sh:** A script to initialize and run services within the Docker container.

## Directory Structure

```
docker/
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
└── .dockerignore
```

## Prerequisites

Before proceeding, ensure you have the following installed on your system:

- **Docker:** [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose:** [Install Docker Compose](https://docs.docker.com/compose/install/)
- **Git:** For version control and repository cloning

## Building Docker Images

To build the Docker images required for the project, navigate to the `docker/` directory and execute the build command.

```bash
cd docker/
docker build -t dynamic_nn_refinement:latest .
```

**Explanation:**

- `-t dynamic_nn_refinement:latest`: Tags the Docker image with the name `dynamic_nn_refinement` and the tag `latest`.
- `.`: Specifies the build context as the current directory.

## Running Services with Docker Compose

Docker Compose simplifies the process of managing multi-container Docker applications. The `docker-compose.yml` file defines the services, networks, and volumes required.

### Start All Services

To start all services defined in the `docker-compose.yml` file:

```bash
docker-compose up -d
```

**Explanation:**

- `up`: Builds, (re)creates, starts, and attaches to containers for a service.
- `-d`: Runs containers in the background (detached mode).

### Services Included

1. **API Server:**  
   - **Image:** `dynamic_nn_refinement:latest`  
   - **Ports:** `8000:8000`  
   - **Description:** Hosts the FastAPI application for serving model predictions.

2. **Prometheus:**  
   - **Image:** `prom/prometheus:latest`  
   - **Ports:** `9090:9090`  
   - **Description:** Collects and stores metrics from the application for monitoring.

3. **Grafana:**  
   - **Image:** `grafana/grafana:latest`  
   - **Ports:** `3000:3000`  
   - **Description:** Visualizes metrics collected by Prometheus through customizable dashboards.

## Managing Containers

### Viewing Running Containers

To view all running containers:

```bash
docker-compose ps
```

### Stopping Services

To stop all running services:

```bash
docker-compose down
```

**Explanation:**

- `down`: Stops containers and removes containers, networks, volumes, and images created by `up`.

### Rebuilding Services

If you make changes to the Dockerfile or application code, rebuild the images:

```bash
docker-compose up -d --build
```

**Explanation:**

- `--build`: Forces the rebuild of images before starting containers.

## Environment Variables

The `docker-compose.yml` file references environment variables defined in the `.env` file located in the root directory of the project. Ensure that all necessary variables are set to allow seamless integration between services.

### Example `.env` File

```env
# .env

# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Prometheus Configuration
PROMETHEUS_HOST=prometheus
PROMETHEUS_PORT=9090

# Grafana Configuration
GRAFANA_HOST=grafana
GRAFANA_PORT=3000
```

**Note:**  
Sensitive information such as API keys or database credentials should be securely managed and not hard-coded.

## Troubleshooting

### Common Issues

1. **Port Conflicts:**  
   If a port is already in use, Docker Compose will fail to start the service.  
   **Solution:** Change the port mapping in `docker-compose.yml` or free up the port.

2. **Image Build Failures:**  
   Errors during the `docker build` process.  
   **Solution:** Check the `Dockerfile` for syntax errors and ensure all dependencies are correctly specified.

3. **Service Not Accessible:**  
   After starting services, if you cannot access them via the browser or API calls.  
   **Solution:**  
   - Verify that the containers are running using `docker-compose ps`.  
   - Check firewall settings that might block the ports.  
   - Review logs for any runtime errors:
     
     ```bash
     docker-compose logs <service_name>
     ```

### Viewing Logs

To view logs for a specific service:

```bash
docker-compose logs <service_name>
```

Replace `<service_name>` with `api`, `prometheus`, or `grafana` as defined in your `docker-compose.yml`.

### Example:

```bash
docker-compose logs api
```

## Contributing

Contributions to the Docker configurations are welcome! To contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/update-docker-setup
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "docs: update Docker configuration for enhanced scalability"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/update-docker-setup
   ```

5. **Open a Pull Request**

   Submit a pull request with a detailed description of your changes and their benefits.

For detailed guidelines, refer to the [Best Practices](../docs/best_practices.md) documentation.

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](../LICENSE).

## Contact

For questions, suggestions, or support, please open an issue on the [GitHub repository](https://github.com/redx94/Dynamic-Neural-Network-Refinement/issues) or contact the maintainer at [qtt@null.net](mailto:qtt@null.net).

---
