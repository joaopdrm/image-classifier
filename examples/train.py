from utils.train import train
from models.architecture import cnn

model = cnn()

data_dir = ""
validation_split = 0.2
batch_size = 32
image_size = (300, 250)

train(data_dir=data_dir,validation_split=validation_split,model=model,batch_size_=batch_size,image_size=image_size)