import torch

#NOTE: This programme is supposed to simulate a biological process, therefore we don't want all the cells to be 
#      updated with each iteration, which would kind of imply that there is this global clock and with each iteration 
#      everybody updates. What we waint its for this process to be more or less random.

class CAModel(torch.nn.Module):
    """Cell automata model.

    Parameters
    ----------
    n_channels : int
        Number of channels of the grid.

    hidden_channels : int
        Hidden channels that are related to the pixelwise 1x1 convolution.

    fire_rate : float
        Number between 0 and 1. The lower it is the more likely it is for
        cells to be set to zero during the `stochastic_update` process.

    device : torch.device
        Determines on what device we perfrom all the computations.

    Attributes
    ----------
    update_module : nn.Sequential
        The only part of the network containing trainable parameters. Composed
        of 1x1 convolution, ReLu and 1x1 convolution.

    filters : torch.Tensor
        Constant tensor of shape `(3 * n_channels, 1, 3, 3)`.
    """
## CONSTRUCTOR ##
    def __init__(self, n_channels=16, hidden_channels=128, fire_rate=0.5, device=None):
        super().__init__() #call constructor of parent


        self.fire_rate = 0.5
        self.n_channels = n_channels
        self.device = device or torch.device("cpu") #if user doesn't specify device we will default to cpu

    ## Perceive step (which is just a 3x3 convolution)
        #sobel filter = does approximation of the gradient (to tell our current cell what is happening around it and in
        #               what direction it would need to go to maximize or minimize the intensity)
        sobel_filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scalar = 8.0

        sobel_filter_x = sobel_filter_ / scalar
        sobel_filter_y = sobel_filter_.t() / scalar
        #identity filter = if we slide this filter over any image, we will get exactly the same image
        identity_filter = torch.tensor(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                dtype=torch.float32,
        )
        #we take the three filters that we defined and stack them together along the 0-th dimension
        #(the ultimate goal is to take these three filters and apply them to each of the channels of the input image,
        # therefore we would end up with a new image that will have three times as many channels)
        filters = torch.stack(
                [identity_filter, sobel_filter_x, sobel_filter_y]
        )  # (3, 3, 3)
        #here we just repeat the filters over all channels and send them to the right device
        filters = filters.repeat((n_channels, 1, 1))  # (3 * n_channels, 3, 3)
        #and finally we store them internally as an attribute because we will use them in the forward pass
        #THESE FILTERS ARE NOT LEARNABLE!!! (we manually hardcoded them)
        self.filters = filters[:, None, ...].to(
                self.device
        )  # (3 * n_channels, 1, 3, 3)

    ## Update step
        #THIS IS THE ONLY PLACE WHERE WE WILL HAVE TRAINABLE PARAMETERS
        #we use the sequential model to define three consecutive steps.
        self.update_module = torch.nn.Sequential(
                #We apply the 1x1 convolution,
                torch.nn.Conv2d(
                    3 * n_channels,
                    hidden_channels,
                    kernel_size=1,  # (1, 1)
                ),
                #then the ReLU activation,
                torch.nn.ReLU(),
                #and finally again another 1x1 convolution
                torch.nn.Conv2d(
                    hidden_channels,
                    n_channels,
                    kernel_size=1,
                    bias=False,
                ),
        )

        #our seed starting image is going to be a single bright pixel in the middle of the image. All the other 
        #pixels (cells) will be non-active and by adjusting the weight and the bias of this second 1x1 convolution, 
        #we're making sure it will actually take a couple of iterations of this rule to populate the pixels that 
        #are further away from the center. The main motivation behind this is to make the training simpler and make 
        #sure we don't end up with a very complicated pattern just after the first iteration
        with torch.no_grad():
            self.update_module[2].weight.zero_()

        #we recursively send all the parameters of this module to our desired device
        self.to(self.device)

## SOME HELPFUL METHODS ##
    def perceive(self, x):
        """Approximate channelwise gradient and combine with the input.

        This is the only place where we include information on the
        neighboring cells. However, we are not using any learnable
        parameters here.

        The goal of the perceive step is to look at the surrounding pixels 
        (or, cells) and understand how the intensity changes.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, 3 * n_channels, grid_size, grid_size)`.
        """
        #we take the filters we prepared in the constructor and perform a so-called depth-wise convolution (by 
        #setting groups equal to the number of input channels)
        return torch.nn.functional.conv2d(x, self.filters, padding=1, groups=self.n_channels)

    def update(self, x):
        """Perform update.

        Note that this is the only part of the forward pass that uses
        trainable parameters (and it's exactly those parameters inside 
        the 1x1 convolution layers).

        Paramters
        ---------
        x : torch.Tensor
            Shape `(n_samples, 3 * n_channels, grid_size, grid_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
        """
        #we prepared everything in the constructor so
        return self.update_module(x)

    @staticmethod
    def stochastic_update(x, fire_rate):
        """Run pixel-wise dropout.

        Unlike dropout there is no scaling taking place.

        We are not actually scaling the remaining values by any scalar.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.

        fire_rate : float
            Number between 0 and 1. The higher the more likely a given cell
            updates.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
        """
        device = x.device

        #we create a boolean mask for each pixel and then we just element-wise multiply the original tensor with 
        #the mask. Note that this mask is going to be broadcasted over all the channels, so it cannot happen that 
        #some channels of given pixels are active and the remaining ones are inactive
        mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(device, torch.float32)
        return x * mask  # broadcasted over all channels

    @staticmethod
    def get_living_mask(x):
        """Identify living cells.

        Takes the alpha channel of our image, which will be the 
        fourth one, and it will use it to determine whether a given 
        cell is alive or not.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, 1, grid_size, grid_size)` and the
            dtype is bool.
        """
        #if the cell itself or any cell in the neighbourhood has an alpha channel higher than 0.1, this cell will
        #be considered as alive
        return (
            torch.nn.functional.max_pool2d(
                x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1
            )
            > 0.1
        )

    def forward(self, x):
        """Run the forward pass.

        Calling the forward method once in our case will mean nothing 
        else than one iteration of the rule. What we will actually do
        while training is to call the forward method multiple times to 
        simulate multiple iterations.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_sample, n_channels, grid_size, grid_size)`.
        """
        #pre-life mask = tensor of booleans
        pre_life_mask = self.get_living_mask(x)

        #we take our input tensor and run the perceived step, which applies the identity and the two Sobel filters
        y = self.perceive(x)
        #we run the update step that contains learnable parameters
        dx = self.update(y)
        #we run the stochastic update, whose goal is to ensure some cells don't get updated during this forward 
        #pass and thus making it more biologically plausible
        dx = self.stochastic_update(dx, fire_rate=self.fire_rate)

        #here we actually use a residual block, which is very important because the new image is nothing else than 
        #the previous image plus some delta image. Here, one can make the argument of we will run this forward method 
        #multiple times and one way to think about this is that you're just creating a very deep architecture
        x = x + dx

        #we compute the post-life mask
        post_life_mask = self.get_living_mask(x)
        #the final life mask is going to be an element-wise AND operator between the pre-life mask and the post-life mask
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)

        return x * life_mask
