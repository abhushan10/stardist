import tensorflow as tf
import zipfile

# # extract saved model directory from zip file
# with zipfile.ZipFile('TF_SavedModel.zip', 'r') as zip_ref:
#     zip_ref.extractall()

# provide the path to the saved model directory
saved_model_dir = "./TF_SavedModel"

# load the saved model
loaded_model = tf.saved_model.load(saved_model_dir)

# print the TensorFlow version used to save the model
print("TensorFlow version of the saved model: ", loaded_model.tensorflow_version)