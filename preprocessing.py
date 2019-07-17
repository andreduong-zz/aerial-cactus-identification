# specify batch_size
batch_size = 150

# create data generator for the training data
train_datagen = ImageDataGenerator(rescale = 1./255,         # normalize pixel values to [0,1]
                                   rotation_range = 30,      # randomly applies rotations
                                   width_shift_range = 0.3,  # randomly applies width shifting
                                   height_shift_range = 0.3, # randomly applies height shifting
                                   horizontal_flip = True,   # randonly flips the image
                                   fill_mode = 'nearest')    # uses the fill mode nearest to fill gaps created

# create data generator for the test data
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe=train[:15001],
                                                    directory=train_dir,
                                                    x_col='id', y_col='has_cactus',
                                                    class_mode='binary',
                                                    batch_size=batch_size,
                                                    target_size=(150,150))

validation_generator = validation_datagen.flow_from_dataframe(dataframe=train[15000:],
                                                              directory=train_dir,
                                                              x_col='id', y_col='has_cactus',
                                                              class_mode='binary',
                                                              batch_size=50,
                                                              target_size=(150,150))