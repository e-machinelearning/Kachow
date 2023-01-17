# Kachow
## Capstone 3 Final Report

The suggestion of this project came from a customer by the name of Jane Doe who happens to have a 7 year old child named Ralph. Ralph recently began his first semester of 2nd grade and is brimming with excitement at meeting his friends, new and old! Jane is also excited because the school offers a free bus system that passed by her house so she doesn't have to waste time on gas, which can be very expensive these days. Now while this is great, the only problem seems to be a crosswalk that the kids have to cross at school in order to get to the bus. Not only that but that stop sign at the crosswalk is covered by a lot of tree brush making it hard for oncoming cars to see. Usually this would be handled by a crosswalk safety guard but Jane as a worried mother wants to be extra safe. 

### Problem Statement

She's come up with an idea to create a bright flashing STOP and SLOW signs to make it easier for the oncoming traffic to see. The system would theoretically take a picture of the oncoming traffic and then once it classifies as a vehicle it would begin flashing bright red and yellow signs to let the traffic know. Our job is to create the classification system that she will be using! 

### Data Loading/Wrangling

So what we're dealing with here is a computer vision problem! It's not a usual machine learning problem of using linear or logistic regression or K-Means clustering in order to get a prediction or classification of data. We're dealing with a deep learning problem that uses neural networks in order to create a well trained model. Since we're dealing with images it's not as simple as reading in a csv file. We had to first go into our directory where our images are contained and then load them individually into lists, one for the images and another for the labels of the images. I accomplished this through a nested for loop that looped through my categories (vehicle or non-vehicle) and then looped through each image individually. And what came out of the first image was this!

![image](https://user-images.githubusercontent.com/99514228/212940569-5ffdbbef-a510-4474-8a12-e51356d7701d.png)

Now as humans we can tell that this isn't a vehicle but for a machine it can't really tell unless we specifically tell it that it is a non-vehicle. How we teach a machine to learn the difference on it's own is through numbers! Machines are really good with numbers and luckily pixels are actually just a bunch of numbers in a specified order. But machines aren't miracle workers, we have to make some changes to the data before we perform the model training.

#### Resizing

The first part of changes we have to make is resizing the data. Resizing is important as it will be the clarity of the image. Like humans, machines can also have a hard time telling things through blurriness. If things are too blurry then it will have a hard time to tell things like edges of where the car begins or ends. Since our first image was a little blurry, we resized it up to 112 which was almost double from the original size of 65 to help improve clarity and also to standardize all the image sizes incase there are some that were different size. 

#### Scaling

Next, like with most ML or DL problems, we have to scale the data. Scaling the data is pretty easy for pixels. Computers store pixels in powers of 2 bytes and pixels happened to be assigned 8 bits for that. So in simpler terms that means pixels are 2^8 or 256 or 0-255 since pixels can be in decimal ranges. What we can do is take all our images while loading them and then divide them by each 255 to make sure they're all scaled properly with each other.

#### Shuffling

Once our data is all loaded, resized, and scaled we move onto shuffling the data! When we plug this through a model if things are not properly shuffled it can create a bad learning environment for the machine. For example when we loaded the data it's all loaded with the non-vehicle images first and then vehicle images second. So what ends up happening is the model learns really fast what a non-vehicle is and then ends up only guessing non-vehicle for the images. So we have to mix the data up a bit so it's randomized what the next image the model will get and so it get's to learn both vehicle and non-vehicle. This is pretty simple, all we have to do is take our images and labels, zip them up together, randomly shuffle them with built in python function, and them unzip them back into there respective lists. 

#### Label Encoding

What we deal with now is label encoding. Right now our labels of the images are in the form of strings. Like I said earlier, machines are very good with numbers but not so much with strings. So in order to get the machine to correctly categorize the images we have to change the labels into numbers. In this case it's 0 and 1. 0 being a non-vehicle, and 1 being a vehicle. We can add more numbers for classification but for this specific project we only need 2 since we are only classifying between those 2 things. 

#### Reshaping

For the final part of our wrangling we have to reshape our images. This might be confusing, reshaping, resizing, and scaling might all sound the same to you but they are different! CNN models take in a certain shape of arrays and so when we reshape it we reshape it with the image size that we set which in our case is 112 and then alongside that in the reshaping parameters we input a 3 since we have color images and color comes in the form of the number 3 like RGB. From there we finished our data wrangling. 

### Data Visualization

Now we can do some data visualization. It's important that when you make changes to your data that you always get a look at it afterwards to check if everything looks okay and no mistakes happened in the process. So what we do next is we take a random 36 images from our lists of images and labels and wee plot them in a grid-like fashion! 

![image](https://user-images.githubusercontent.com/99514228/212951347-e4294cd1-dac0-4d9a-b493-0743e00212cd.png)

We can see that our images are a lot more clearer than before from the resizing and overall look pretty good! Luckily there wasn't much we had to do in this portion of the project but the importance of this step is making sure everything is ready to be put in through our model which we will create as our next step. 

### Pre-Processing and Training

#### Splits

Usually when we work with ML models it's common that we 2 sets of data, training and testing. With CNN I recommend using 3 sets: training, validation, and testing. Training which will be used to train the data, a validation set which will be used to compare the training set to while it is being trained, and a testing set to test the model on with unseen data. Validation is added because it acts as a sort of mini-testing set to help fine tune our model during the training process. We split our data into 70% training, 20% validation, and 10% testing. 

#### Callbacks

When creating a CNN model, it can take a long time to train a model sometimes up to weeks of training depending on the problem and the data you're handling. Our model isn't going to take that long but it's still good practice to create callbacks. Callbacks help to monitor and optimize the model during the training so we can see what is happening. For our callbacks we added a tensorboard log to record all our training in a log, an early stopping callback to stop our model when it is no longer improving, and a learning rate scheduler to modify our model's learning rate during training so we don't have a high learning rate towards the learning process creating overfitting. 

#### CNN Model Architecture

With our model we are going with a basic Sequential model which is standard for most CNN model cases. Sequential models are simple but work very well. The architecture goes as follows: 

The Conv2D deals with our input images which are seen as 2D matrices and takes them in. It creates the filter and kernel size at which the model will look at in the image arrays.

MaxPool2D downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window for each channel of the input.

Batch Normalization normalizes the contributions to a layer for every mini-batch which decreases the number of epochs we need to train our model and helps prevent overfitting.

Flatten will connect our convolution layers to our Dense layers which is the standard layer type used in CNN models.

We are basically going to have 4 layers for the network and the activation function we will be using is relu and sigmoid which are also standard functions.

With our architecture complete we can compile the model! For our compiling we will use binary crossentropy since our model is binary classification meaning we classify between 2 things which in our case is vehicles or non-vehicles. We will stick with an adam optimizer that will help train our data with fewer resources and is very adaptive compared to other optimizers. It is an alternative optimization algorithm that provides more efficient neural network weights by running repeated cycles of “adaptive moment estimation.”

The metrics we will be using is accuracy and loss and for now we will go with 4 epochs to train our data since we have a quite a bit of layers in our CNN model architecture. We used 4 but most basic models have only 2-3. We compile the model and put our data through it and around 30min to an hour later we get our results!. 

#### Training Results

At the end of the training process we ended up with 99.5% training accuracy and a 97.44% validation accuracy which is great. When evaluated on the testing set it ended up with a 97.92% accuracy rate. The accuracy and loss graphs also showed very good results: 

![image](https://user-images.githubusercontent.com/99514228/212962204-bb70aff1-d510-4b21-9f9b-60aec0bb64b6.png)

![image](https://user-images.githubusercontent.com/99514228/212962237-4fd55b56-3769-4a95-b4bc-dcc9e6da1783.png)

Both accuracy and loss sets stay below the training set but follow along the curve of the training set which is what we look for in a good model. All that is left for us is to test our model on our testing set!

### Modeling

For our modeling portion we used the predict function to first predict across the testing set. From there we looped through the predictions and using else/if we appended whether or not the model predicted a vehicle or non-vehicle and compared it to the actual labels of the testing set. Out of the initial 20 that we tested we ended up with only 1 wrong which is to be expected since our model is not meant to be perfect but showed great results nonetheless! 

Using a similar method from our data visualization, we took a random 36 images from our test set and used our model to predict whether or not the images were vehicle or non-vehicle and this time our model correctly predicted them all correctly!

![image](https://user-images.githubusercontent.com/99514228/212976492-e129ff2e-673d-4b89-8e4b-2a0f312d1d10.png)


### Conclusion

With that we were able to save our model and safely hand it over to Jane Doe to use it in her crosswalk safety project. We recommend as a next step that she create a vehicle detection (not classification!) model to be used in combination with our model in order to detect cars in real time video instead of through an image!


