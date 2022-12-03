class CFG:
#     wandb=True
    wandb=False
    competition='FB3'
    _wandb_kernel='wallykop'
    debug=False
    apex=True
    print_freq=20
    num_workers=4
    model="/kaggle/input/debertav3base/"
    gradient_checkpointing=True
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=5
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=8
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=2807
    n_fold=4
    trn_fold=[0, 1, 2, 3]
    train=True