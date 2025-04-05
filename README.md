# MLSimplified

A machine learning library that makes training and using ML models simple.

## Installation

```bash
pip install mlsimplified
```

## Features

- Automatic problem type detection (classification/regression)
- Smart data preprocessing + cleaning
- Auto model selection
- One line training and prediction
- Automatic evaluation and reporting
- Visualization
- Error handling
- Efficient memory management

## Usage

### Basic Usage

```python
import pandas as pd
import numpy as np
from mlsimplified import Model

'''DATA'''
train_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 100),
    'feature2': np.random.normal(0, 1, 100),
    'target_column': np.random.rand(100)  # Regression example
})

test_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 20),
    'feature2': np.random.normal(0, 1, 20),
})


"""The Actual Model"""

# Initialize model
model = Model(train_data, target="target_column")

# Train the model
model.train()
model.summary()
model.report()

# Make predictions
predictions = model.predict(test_data)

# Export the model
model.export("model.pkl")
```

### Advanced Usage

```python
import pandas as pd
import numpy as np
from mlsimplified import Model

# Create sample data with categorical and numerical features
data = pd.DataFrame({
    'numeric_feature': np.random.normal(0, 1, 100),
    'categorical_feature': np.random.choice(['A', 'B', 'C'], size=100),
    'label': np.random.choice([0, 1], size=100)
})

# Initialize with custom parameters
model = Model(data, target="label")

# Train with custom test size
model.train(test_size=0.3)

# Get model insights
model.summary()  # View model information
model.plot()     # Visualize feature importance
model.report()   # Generate evaluation report
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