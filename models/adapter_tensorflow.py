from adapter import Modeladapter
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import logging
import warnings
import os
import datetime
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

class Adaptertensorflow(Modeladapter):

    def __init__(self,model: tf.keras.Model):

        self.model = model

    def train(self,train_ds:np.ndarray,val_ds:np.ndarray,X:np.ndarray,Y:np.ndarray,
        shuffle:bool=False,X_t:np.ndarray=None,y_t:np.ndarray=None,model_type:str=None,**kwargs):

        """
        Esta função é responsável pelo treinamento de modelos do Keras, esta função tem como
        default 30 epocas e tamanho 64 de batch_size, caso queira passar números diferentes para 
        o treinamento basta seguir este exemplo:

        ***model.train(X=X, Y=Y, X_t=X_t, y_t=y_t,filepath=filepath ,epochs=5, batch_size=12)***

        Parâmtros:
        ---------

        X->np.ndarray: Argumento retornado pelo data_prep.py
        Y->np.ndarray: Argumento retornado pelo data_prep.py
        X_t->np.ndarray: Argumento retornado pelo data_prep.py
        y_t->np.ndarray: Argumento retornado pelo data_prep.py

        filepath->str: caminho para pasta onde o modelo será salvo
        """

        self.model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "accuracy"])

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="/tmp/checkpoint",
        monitor="accuracy",
        mode='max',
        save_weights_only=True,
        save_best_only=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0,
        )

        tensorboard_callback = TensorBoard(
        log_dir="logs",
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        update_freq="epoch",
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None,
        )
        
        epochs = kwargs.get("epochs", 60)  
        batch_size = kwargs.get("batch_size", 32)

        if model_type == 'image':
            self.model.fit(train_ds, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                           callbacks=[model_checkpoint_callback, early_stopping, tensorboard_callback],
                           validation_data=val_ds)
        else:
            self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                           callbacks=[model_checkpoint_callback, early_stopping, tensorboard_callback],
                           validation_data=(X_t, y_t) if (X_t is not None and y_t is not None) else None)


    def test(self,X_v:int):

        """
        Parâmetros:
        ----------

        model: arquitetura LSTM instanciada a partir do script architecture.py
        """

        self.model.load_weights('/tmp/checkpoint')
        y_pred = self.model.predict(X_v, verbose=1)
        predictions = (y_pred > 0.5).astype("int32")
        return predictions,y_pred
    
    def save(self, path_name:str):

        """
        Esta função está responsável por salvar o modelo treinado

        Parâmetros:
        ----------
        path_name->str: nome da pasta onde o modelo será salvo
        """

        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        volta_um_diretorio = os.path.dirname(diretorio_atual)
        output_directory = os.path.join(volta_um_diretorio, "tensorflow_models")

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        experiment_directory = f"{path_name}-" + datetime.datetime.now().strftime("%Y-%m-%d")
        experiment_directory = os.path.join(output_directory, experiment_directory)

        os.makedirs(experiment_directory)

        model_dir = os.path.join(experiment_directory, 'model.h5')

        self.model.save(model_dir)

        logging.info(f' modelo do tipo keras salvo no diretorio {model_dir}')