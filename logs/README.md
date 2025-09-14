# Logs Directory

This directory contains application logs, training logs, and monitoring data.

## Log Files Structure

```
logs/
├── training/                   # Training session logs
│   ├── model_training.log     # Model training progress
│   ├── optimization.log       # Threshold optimization logs
│   └── evaluation.log         # Model evaluation logs
├── application/               # Web application logs
│   ├── streamlit.log         # Streamlit app logs
│   └── prediction.log        # Prediction request logs
└── monitoring/               # Performance monitoring
    ├── model_performance.log  # Model performance metrics
    └── system_metrics.log     # System resource usage
```

## Log Rotation

- Logs are rotated daily to prevent excessive disk usage
- Archive logs are compressed and stored for 30 days
- Critical errors are immediately logged to separate error files

## Monitoring

- Model performance is continuously monitored
- False alarm rates are tracked
- System resource usage is logged for optimization