from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader

@dataclass
class Trainer:
    # Batch config
    """
    During training, the following high-level process takes place. The model 
    iterates over each sample in the training dataset, predicts the output, and 
    updates its parameters at some point, not quite sure at what interval this
    takes place though.

    A batch contains all or a portion of the training dataset. For example, a 
    training dataset of 1k could be configured in one batch of 1k samples, two
    batches of 500 samples, and so on. 

    An epoch is one complete pass over the training data, containing one or more
    batches, as configured above. The number of epochs is a hyperparameter 
    controlled by the user. Training the model too much could lead to 
    overfitting the model causing the model to perform unwell on new, real-world
    data.
    
    """
    batches_per_epoch: int

    """
    Tokens are a unit of data processed by the model during training and 
    inference that assist the LLMs given task. e.g., generation, reasoning, and
    prediction.

    Each batch contains N number of tokens to train on. This number is a 
    hyperparameter and controlled by the user because the hardware available 
    for training will determine how many tokens can be seen during batch 
    processing because all tokens will be loaded into memory. 

    Controlling the tokens per batch makes it easier to manage the memory and
    compute rather than processing batches that vary wildly in token length 
    from one batch to another. 

    This ensures the GPU is constantly used to its maximum capacity and that
    out of memory issues never happen with CUDA. 
     
    """
    tokens_per_batch: int

    # Epoch config
    """
    The number of times to train on the training dataset

    Note: In Molmo's fit function, when certain criteria are met the training
    dataset gets shuffled. I am assuming this takes place so the next epoch does
    not see identical batches.
    """
    epoch: int

    """
    Maximum number of times to train on the training dataset.
    
    Warning: Doing so too many times will result in overfitting on the data.

    """
    max_epochs: int

    """
    During training, a step will update the models parameters, aka a gradient
    update. Not enough updates and the cost function is not optimized as much as
    it could be.
    """
    max_steps: int

    """Number of steps throughout the entire training process"""
    global_steps: int

    """
    Tracks the global number of training examples seen in the current epoch 
    for the purpose of restoring the data loader position on restarts.

    Note: Directly from Molmo
    
    """
    global_train_examples_seen_this_epoch: int

    """
    Tracks the global total number of tokens trained on.

    Note: Directly from Molmo
    """
    global_train_tokens_seen: int

    # Token config
    """Maximum number of tokens to be seen during training"""
    max_tokens: int

    # Checkpoint info

    """
    Checkpointing is a technique used during training of a machine learning 
    model that stores the latest version of the model on a set interval. That
    was just in case training is interrupted, it can resume from the last 
    successful checkpoint, it could also be used to store the last and best
    model.
    """

    """Directories where to save the checkpoints"""
    checkpoints: list[Path]

    """
    Temporary checkpoints that are meant to be used if training is interrupted.
    """
    ephemeral_checkpoints: int

    """
    Stored the model across multiple files.
    
    This is useful for when training models that are too large to store on one
    device. This is the case with Molmo, and they leverage PyTorch's FSDP
    class to manage this.
    """
    last_sharded_checkpoint_step: int

    """
    The last step we saved the unsharded model.
    """
    last_unsharded_checkpoint_step: int

    """
    Contains the full model in a single, consolidated format.

    Great for inference, transferring to another framework and finetuning.
    
    """
    unsharded_checkpoints: list[int]
    
    # Training config
    """
    Specify the details of the entire training process.

    Molmo supports single and multitask training configurations.

    More details here: https://deepwiki.com/allenai/molmo/3.3-training-configuration
    
    """
    training_config: TrainConfig

    """
    PyTorch DataLoader that handles loading the training dataset, which handles
    multiprocessing, shuffling, etc.

    Note: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    train_loader: DataLoader

    """
    The current training loss.

    How well the model is performing on the training dataset, i.e., based on the
    input are we getting the expected output.

    Question: For generation tasks where no one response is correct, how is the
    training loss calculated for these tasks ?
    """
    curr_train_loss: int
    
    # Training schedule

    """
    ML models are trying to learn some function during training, i.e, model
    weights that sufficiently solve some problem space. The rate at which these
    models are updated are essential to the eventual inference performance of
    the model. To control this learning rate, a learning rate scheduler is used
    to move the weights in the right direction, at the right pace or slope.
    """
    scheduler: Scheduler

    """
    The unit to control the learning rate, either number of tokens seen or steps
    taken.
    """
    scheduler_current: SchedulerUnits

    """Either the max number of steps or tokens used for training."""
    scheduler_max: int

    # Training metrics
    """
    The loss function to optimize during training, determines how well the model
    is performing on the training dataset.
    """
    loss_fn: int

    """The minimum training loss value found during training"""
    min_train_loss: float

    # Dataset

    """Dataset used for training."""
    dataset: int

    # Device
    """PyTorch: Where the training will take place"""
    device: int
    
    # Evaluation
    """
    Evaluates the model during training.

    There are different metrics to evaluate the model on during training such
    as the CLIP Score, Visual Information Fidelity, or Image Gradients.

    The evaluations available to Molmo all come from Lightning AI's torchmetrics
    package which have all of these metrics implemented in PyTorch.

    Note: https://lightning.ai/docs/torchmetrics/stable/gallery/
    
    """
    evaluators: int

    # Distributed training
    """
    PyTorch: Fully Sharded Data Parallelism used to store model parameters
    on multiple nodes in a compute cluster when one machine is not enough to
    store a model's parameters.
    """
    fsdp_model: int

    # Optimizer
    """
    Responsible for updating the model weights during training. Based on the
    optimizer function and its performance, the learning rate will differ.

    For example, the the Lion optimizer is more computationally efficient than
    the AdamW optimizer so its learning rate can be 3-10x smaller and
    subsequently the weight decay can be 3-10x larger to maintain a similar
    strength.
    """
    optim: int


