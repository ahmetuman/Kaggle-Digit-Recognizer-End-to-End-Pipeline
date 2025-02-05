# Kaggle-Digit-Recognizer-End-to-End-Pipeline

This project's main goal is to deploy a basic digit recognizer (I recently solved in Kaggle) using FastAPI. The most important part of this project is to deploy it in a containerized environment.

## Performance Metrics (Dockerized API)

The following metrics were obtained by running 100 consecutive prediction requests on the containerized API:

### Latency Statistics
| Metric | Value (milliseconds) | Description |
|--------|---------------------|-------------|
| Average | 42.33 ms | Mean response time across all requests. This represents the typical expected latency for any given request. |
| Median | 40.54 ms | The middle value of all response times. Lower than average indicates good consistency with few outliers. |
| 95th Percentile | 43.63 ms | 95% of requests complete within this time. Shows the upper bound for most normal operations. |
| 99th Percentile | 52.73 ms | 99% of requests complete within this time. Represents worst-case scenarios. |

### Analysis
- **Consistency**: The small gap between median (40.54ms) and average (42.33ms) indicates very stable performance with few outliers.
- **Predictability**: 95% of requests complete within 43.63ms, showing highly consistent response times.
- **Worst-case Performance**: Even in the worst 1% of cases (99th percentile), latency remains under 53ms.
- **Production Readiness**: These metrics demonstrate the system's capability to handle real-time prediction requests with reliable response times.