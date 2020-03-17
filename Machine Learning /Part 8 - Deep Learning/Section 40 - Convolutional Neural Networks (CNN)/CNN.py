from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#---------------------------- 1. BUILD CNN MODEL ------------------------------

# Initializing CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), 
                             input_shape = (64, 64, 3), #input_shape - specifiy shape of images
                             activation = 'relu')) 

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding 2nd convolution layer to gain more accuracy
classifier.add(Convolution2D(32, (3, 3), activation = 'relu')) #applied on pooled feature map coming from prv layer instead of images
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#----------------------- 2. FITTING CNN To THE IMAGES -------------------------

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,
                                   zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory("dataset/training_set",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = train_datagen.flow_from_directory('dataset/test_set',
                                             target_size = (64, 64),
                                             batch_size = 32,
                                             class_mode = 'binary')

classifier.fit_generator(training_set, steps_per_epoch = 8000,
                         epochs = 10, validation_data = test_set,
                         validation_steps = 2000)
