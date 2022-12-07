# Common training-related configs that are designed for "tools/train_net.py"
# You can use your own instead, together with your own train_net.py
train = dict(
    # Directory where output files are written to
    output_dir="./output",
    # The initialize checkpoint to be loaded
    init_checkpoint="",
    # The total training iterations
    max_iter=90000,
    # options for Automatic Mixed Precision
    amp=dict(enabled=False),
    # options for DistributedDataParallel
    ddp=dict(
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    # options for Gradient Clipping during training
    clip_grad=dict(
        enabled=False,
        params=dict(
            max_norm=0.1,
            norm_type=2,
        ),
    ),
    # options for Fast Debugging
    fast_dev_run=dict(enabled=False),
    # options for PeriodicCheckpointer, which saves a model checkpoint
    # after every `checkpointer.period` iterations,
    # and only `checkpointer.max_to_keep` number of checkpoint will be kept.
    checkpointer=dict(period=5000, max_to_keep=100),
    # Run evaluation after every `eval_period` number of iterations
    eval_period=5000,
    # Output log to console every `log_period` number of iterations.
    log_period=20,

    # wandb logging params
    wandb=dict(
        enabled=False,
        params=dict(
            dir="./wandb_output",
            project="detrex",
            name="detrex_experiment",
        )
    ),

    device="cuda",
    # ...
)
