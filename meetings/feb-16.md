# Feb 16
Location: zoom

### Agenda:
- Plan moving forward for quantum image processing
  
### Notes
- We will be using the MNIST dataset. I will upload it to our repo under a `data` folder in `src`.
- We will need to preprocess the images. Specifically, we will need to reduce the dimensionality of the images to make them easier to work with. This task will begin by looking at resizing with simple premade tools. Then look at more advanced techniques like PCA and preferably deep convolutional auoencoders. The one constraint is that I want the output to still be a 2D image. You could also look into quantum autoencoders down the line.
- The other main task is building a quanvolutional neural net. We will base our work on [this paper [1]](https://arxiv.org/pdf/1904.04767.pdf) and [this paper[2]](https://arxiv.org/pdf/2106.07327.pdf). This will require implementing a quanvolution layer function according to [1] as well as a parameterized quanvolutional layer from [2].

### Action Items
- *Robbie* I want you looking at the preprocssing stage.
- *Julia* you will be looking at the first step of the quanvolutional layer (implementing [1])
- *Tristan* I want you looking into how we will parameterize the layer so it can learn ([2])
- *Connor* I want you to work on orquestrating the overall workflow. Take a look at [1] and [2]. The main task will be combining the quantum and classical layers into one neural network object that can be trained and tested.



