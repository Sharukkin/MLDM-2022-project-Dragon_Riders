model_path = "/kaggle/input/debertav3base/"

class CFG:
    pl_labels = None
    wandb = False
    wandb_alert_freq = "every"  
    competition = "FB3"  
    debug = True
    debug_train_size = 200
    apex = True
    print_freq = 50
    num_workers = 2
    tokenizer =  model_path 
    model = model_path  
    ckpt_name = model_path  
    gradient_checkpointing = True 
    batch_scheduler = True
    scheduler = "cosine"
    num_cycles = 0.5
    use_8bit_optimizer = True
    num_warmup_steps = 50
    epochs = 3
    encoder_lr = 1.5e-5
    layerwise_learning_rate_decay = 0.7225 
    decoder_lr = 1.5e-5 
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 6
    infer_batch_size = 8
    gradient_accumulation_steps=1
    max_len = 640
    window_size = 511  
    edge_len = 32  
    weight_decay = 0.01
    decoder_weight_decay = 2
    max_grad_norm = 1000 
    mlm_ratio = False
    layer_reinitialize_n = 1 
    freeze_n_layers = 0 
    target_cols = [
        "cohesion",
        "syntax",
        "vocabulary",
        "phraseology",
        "grammar",
        "conventions",
    ]
    seed = 2807 
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    train = True
    eval_step_save_start_epoch = 1
    n_eval_steps = int(
        3910 * (n_fold - 1) / n_fold / batch_size * 0.201  # high freq validation
    )
    multi_sample_dropouts = None  # [0.1, 0.2, 0.3, 0.4, 0.5]
