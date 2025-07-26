> Jul 24,25

learn from ZHB's project. Study his file architecture, and understand how modules function and cooperate.

here is a review of his project(with some folders I add):

1. datasets: get dataset and generate dataloaders for cifar10,etc
2. samplers: generating samples of standard guassian noise/ corrupted image/ lognorm times
3. paths: the conceptual/ideal path for the image transformation process
4. models&vectorfields: neural networks with Unet Architecture / other pretrained model as backbone
5. trainer: define the loss function and deal with all data generated in the training process
6. simulators: eulars and heun simulators for image generation|restoration
7. utils: codes for utils functions (visualization + model_utils)  
8. data: the raw data for training
9. checkpoints: save the models params and other key information
10. results : save the image outputs

Plus, I do some modification :

- [x] some minor format improvements and additional comments
- [x] change the get_dataloader functions in the datasets package
- [x] change the trainer function: add param init_channel.
- [x] modify the save_checkpoints function to dump the model in checkpoints
- [x] modify the visualization function   
- [x] modify the path class: use noise(Sampler) and data(DataLoader) sampling a x_t point

Thanks to his project as a reference, it really save me a lot of time in making the project modular and encourage me to carry it forward, achieving more function.

> Jun 26: meanflow mnist/cifar10

I try to understand what's the methods (schduler and early stop...) are used in the **trainer** class in zhb's project. I searched from their role in making the training faster and stable. But finally I select to deprecate the mixed precision because I find it hard to handle (faster though, but the output turned out to be messy)

If time permitted, I will try to learn some mechanism of optimizers and schedulers and scalers. It seems that my partner has already known all of these. He even talked about the mechanism of computing Jacobian matrix under the hood.

more things I accomplished:

- minor changes for switching to mean flow. do a mnist test.
- improve the visualization function. I combine the functions in visual into a generate and save function.
- try transfer learning (use EfficientNet as backbone) and give up.
- learn the mechanism of Adam and SDG: the world of optimization contains to many cases!
- Start to use Muon for the final image restorer.
- review the essay of Mean Flow and Lecture Note from MIT's course.
- Still I am wondering the mathematical principle of Flow Matching: why it works. 

> Jun  

