# import necessary tools
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = load_model("./trained-models/cactus-vgg16.h5")
test_dir = "test"
nrows = 64
ncolumns = 64
columns = 5

test_imgs = ['./test/{}'.format(i) for i in os.listdir(test_dir)] # get test images

def read_and_process_image(list_of_images):
    """
    Returns two arrays: 
        X is an array of resized images
        y is an array of labels
    """
    X = [] # images
    y = [] # labels
    
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))  #Read the image
        # get the labels
        if 'cactus' in image:
            y.append(1)
        elif 'no cactus' in image:
            y.append(0)
    
    return X, y


X_test, y_test = read_and_process_image(test_imgs[0:10]) # Y_test in this case will be empty.
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)

i = 0
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('has cactus')
    else:
        text_labels.append('has no cactus')
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This picture ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break
plt.show()