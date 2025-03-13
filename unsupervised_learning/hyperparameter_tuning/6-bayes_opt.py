import numpy as np
import GPyOpt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data():
    """Load and preprocess the data"""
    # Using Fashion MNIST as an example dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize and reshape the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    
    # Split training data to create validation set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def create_model(params):
    """Create a CNN model with the given hyperparameters"""
    model = models.Sequential([
        layers.Conv2D(
            int(params['filters']), 
            kernel_size=3, 
            activation='relu', 
            input_shape=(28, 28, 1)
        ),
        layers.MaxPooling2D(),
        layers.Dropout(params['dropout_1']),
        
        layers.Conv2D(
            int(params['filters'] * 2), 
            kernel_size=3, 
            activation='relu'
        ),
        layers.MaxPooling2D(),
        layers.Dropout(params['dropout_2']),
        
        layers.Flatten(),
        layers.Dense(
            int(params['dense_units']), 
            activation='relu', 
            kernel_regularizer=tf.keras.regularizers.l2(params['l2_weight'])
        ),
        layers.Dropout(params['dropout_3']),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_checkpoint_name(params):
    """Create a checkpoint filename based on hyperparameters"""
    param_str = '_'.join([
        f"lr_{params['learning_rate']:.6f}",
        f"fu_{int(params['filters'])}",
        f"du_{int(params['dense_units'])}",
        f"l2_{params['l2_weight']:.6f}",
        f"d1_{params['dropout_1']:.2f}",
        f"d2_{params['dropout_2']:.2f}",
        f"d3_{params['dropout_3']:.2f}"
    ])
    return f"checkpoints/model_{param_str}.h5"

def objective_function(x):
    """Objective function to minimize"""
    # Convert the input array to a dictionary of parameters
    params = {
        'learning_rate': float(x[:, 0]),
        'filters': float(x[:, 1]),
        'dense_units': float(x[:, 2]),
        'l2_weight': float(x[:, 3]),
        'dropout_1': float(x[:, 4]),
        'dropout_2': float(x[:, 5]),
        'dropout_3': float(x[:, 6])
    }
    
    # Create and train the model
    model = create_model(params)
    
    # Create checkpoint callback
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = create_checkpoint_name(params)
    checkpoint_callback = callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )
    
    # Early stopping callback
    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint_callback, early_stopping],
        verbose=0
    )
    
    # Return the negative validation accuracy (since GPyOpt minimizes)
    return -np.max(history.history['val_accuracy'])

# Define the parameter space
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'filters', 'type': 'discrete', 'domain': (16, 32, 64)},
    {'name': 'dense_units', 'type': 'discrete', 'domain': (64, 128, 256)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-6, 1e-3)},
    {'name': 'dropout_1', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'dropout_2', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'dropout_3', 'type': 'continuous', 'domain': (0.1, 0.5)}
]

if __name__ == "__main__":
    # Load and prepare the data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    
    # Create and run the Bayesian optimization
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=bounds,
        model_type='GP',
        acquisition_type='EI',
        maximize=False,
        normalize_Y=True,
        initial_design_numdata=5
    )
    
    # Run optimization
    optimizer.run_optimization(max_iter=30)
    
    # Plot convergence
    optimizer.plot_convergence()
    plt.savefig('convergence.png')
    plt.close()
    
    # Save optimization report
    with open('bayes_opt.txt', 'w') as f:
        f.write("Bayesian Optimization Report\n")
        f.write("==========================\n\n")
        f.write(f"Optimization completed at: {datetime.now()}\n\n")
        
        f.write("Best hyperparameters found:\n")
        for i, param in enumerate(bounds):
            f.write(f"{param['name']}: {optimizer.x_opt[i]}\n")
        
        f.write(f"\nBest validation accuracy: {-optimizer.fx_opt:.4f}\n\n")
        
        f.write("Optimization history:\n")
        f.write("Iteration | Validation Accuracy\n")
        f.write("-" * 40 + "\n")
        for i, y in enumerate(optimizer.Y_best):
            f.write(f"{i+1:^9} | {-y:^18.4f}\n")
