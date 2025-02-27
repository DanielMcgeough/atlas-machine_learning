import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder model
    
    Parameters:
        input_dims (int): Dimensions of the model input
        hidden_layers (list): Number of nodes for each hidden layer in the encoder
        latent_dims (int): Dimensions of the latent space representation
        
    Returns:
        encoder: Encoder model outputting latent representation, mean, and log variance
        decoder: Decoder model
        auto: Full autoencoder model
    """
    # Define encoder
    inputs = Input(shape=(input_dims,), name='encoder_input')
    
    # Add hidden layers for encoder
    x = inputs
    for i, nodes in enumerate(hidden_layers):
        x = Dense(nodes, activation='relu', name=f'encoder_hidden_{i}')(x)
    
    # Define latent space parameters
    z_mean = Dense(latent_dims, name='z_mean')(x)  # No activation as specified
    z_log_var = Dense(latent_dims, name='z_log_var')(x)  # No activation as specified
    
    # Sampling function for variational aspect
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dims), mean=0., stddev=1.0)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    # Sampling layer
    z = Lambda(sampling, output_shape=(latent_dims,), name='z')([z_mean, z_log_var])
    
    # Create encoder model
    encoder = Model(inputs, [z, z_mean, z_log_var], name='encoder')
    
    # Define decoder
    decoder_input = Input(shape=(latent_dims,), name='decoder_input')
    
    # Add hidden layers for decoder in reverse order
    y = decoder_input
    for i, nodes in enumerate(reversed(hidden_layers)):
        y = Dense(nodes, activation='relu', name=f'decoder_hidden_{i}')(y)
    
    # Output layer with sigmoid activation
    decoder_output = Dense(input_dims, activation='sigmoid', name='decoder_output')(y)
    
    # Create decoder model
    decoder = Model(decoder_input, decoder_output, name='decoder')
    
    # Create autoencoder
    auto_output = decoder(encoder(inputs)[0])  # Connect encoder and decoder
    auto = Model(inputs, auto_output, name='autoencoder')
    
    # Add VAE loss
    def vae_loss(x, x_decoded_mean):
        reconstruction_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean) * input_dims
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(reconstruction_loss + kl_loss)
    
    # Compile the model
    auto.compile(optimizer='adam', loss=vae_loss)
    
    return encoder, decoder, auto
