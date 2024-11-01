# MLSimplified

A machine learning library that makes training and using ML models simple.

## Installation

```bash
pip install mlsimplified
```

## Quick Start

```python
from mlsimplified import Model

# Create and train a model in one line
model = Model("data.csv", target="label")
model.train().evaluate().predict("test.csv").export("model.pkl")
```

## Features

- Automatic problem type detection (classification/regression)
- Smart data preprocessing and cleaning
- Built-in model selection
- One-line training and prediction
- Automatic evaluation and reporting
- Built-in visualization
- Production-ready error handling
- Efficient memory management

## Usage

### Basic Usage

```python
from mlsimplified import Model

# Initialize with your data
model = Model("your_data.csv", target="target_column")

# Train the model
model.train()

# Make predictions
predictions = model.predict("new_data.csv")

# Export the model
model.export("model.pkl")
```

### Advanced Usage

```python
from mlsimplified import Model

# Initialize with custom parameters
model = Model("data.csv", target="label")

# Train with custom test size
model.train(test_size=0.3)

# Get model insights
model.summary()  # View model information
model.plot()     # Visualize feature importance
model.report()   # Generate evaluation report

# Chain operations
model.train().evaluate().predict("test.csv").export("model.pkl")
```

## API Reference

### Model Class

#### Initialization
```python
Model(data: Union[str, pd.DataFrame], target: str)
```

#### Methods
- `train(test_size: float = 0.2, random_state: int = 42) -> Model`
- `evaluate() -> Model`
- `predict(data: Union[str, pd.DataFrame]) -> np.ndarray`
- `export(path: str) -> Model`
- `summary() -> Model`
- `plot() -> Model`
- `report() -> Model`

## Dependencies

- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- joblib>=1.0.0
- setuptools>=65.5.1

## License

MIT 