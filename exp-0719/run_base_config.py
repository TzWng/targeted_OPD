cfg = Config(
    mode="opd", dataset="deepmath",
    run_prefix="q3-1.7b",
    student="Qwen/Qwen3-1.7B-Base",        
    teacher="/content/drive/MyDrive/OPD/Qwen3-4B",
    dm_diff_min=0.0, dm_diff_max=5.0,
    max_new_tokens=1536, max_prompt_tokens=512,
    temperature=1.0, top_p=1.0, presence_penalty=1.5, enable_thinking=False,
    rollout_is_correction=True, mask_eos_in_loss=True,
    lr=1e-5, warmup_steps=0,
    prompts_per_step=1, grad_accum=4, group_size=4,
    steps=400, save_every=50, eval_every=0, rollout_print_every=10, seed=0)
run(cfg)
