---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:344
- loss:TripletLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Where can I get the Semester Fee Receipt?
  sentences:
  - The fee receipt will be available online for download from the eLearn portal three-weeks
    after the commencement of your classes
  - The fee receipt will be available online for download from the eLearn portal three-weeks
    after the commencement of your classes
  - Please note that it is mandatory for the students to attend either in Regular
    or Makeup Exam for all registered courses in the particular semester. Students
    who miss the Regular or Makeup Exam for any courses during the current semester
    will be declared RRA (Required to Register Again) Grade for that course and they
    have to register for the same course during a later semester. It is crucial to
    note that no further Makeup Exams will be arranged, regardless of the reasons
    for missing the Regular and Makeup Exams.
- source_sentence: How can I update the details of my Supervisor and Additional Examiner
    after Abstract submission?
  sentences:
  - You can resubmit your report once your faculty mentor enables resubmission. Please
    email your professor requesting this. If still issue is not resolved please write
    to support@wilp.bits-pilani.ac.in.
  - ● Employing Organization ● Degree Program ● Research Area ● Dissertation Title
    ● Supervisor's Name ● Supervisor's Email ● Supervisor's Qualification ● Supervisor's
    Designation & Address ● Supervisor's Phone No. (with STD code) ● Additional Examiner's
    Name ● Additional Examiner's Email ● Additional Examiner's Qualification ● Additional
    Examiner's Designation & Address ● Additional Examiner's Phone No. (with STD code)
  - ● Employing Organization ● Degree Program ● Research Area ● Dissertation Title
    ● Supervisor's Name ● Supervisor's Email ● Supervisor's Qualification ● Supervisor's
    Designation & Address ● Supervisor's Phone No. (with STD code) ● Additional Examiner's
    Name ● Additional Examiner's Email ● Additional Examiner's Qualification ● Additional
    Examiner's Designation & Address ● Additional Examiner's Phone No. (with STD code)
- source_sentence: How many pages are required for the Mid-Semester report?
  sentences:
  - There is no fixed page limit for the Mid-Semester report. A sample report will
    be shared via email which you can use as a reference. The focus should be on quality
    and clarity of content rather than the number of pages, ensuring that your faculty
    can effectively evaluate your work.
  - There is no fixed page limit for the Mid-Semester report. A sample report will
    be shared via email which you can use as a reference. The focus should be on quality
    and clarity of content rather than the number of pages, ensuring that your faculty
    can effectively evaluate your work.
  - You can resubmit your report once your faculty mentor enables resubmission. Please
    email your professor requesting this. If still issue is not resolved please write
    to support@wilp.bits-pilani.ac.in.
- source_sentence: How to exit the Programme midway and will I get refund on the Fee
    paid?
  sentences:
  - Please refer to our policy, which is clearly outlined on our website and reiterated
    in the admission offer letter that you have accepted prior to paying the admission
    & semester fee.
  - Please refer to our policy, which is clearly outlined on our website and reiterated
    in the admission offer letter that you have accepted prior to paying the admission
    & semester fee.
  - 'A Supervisor should possess one of the following qualifications: B.E., B.Tech.,
    M.Sc., MBA, or MCA, along with a minimum of 5 years of relevant work experience.'
- source_sentence: How do I register for my previous semester backlog course(s)?
  sentences:
  - Students are allowed to register for a maximum of four courses in one semester.
    If you have any pending backlog courses, the duration of your programme shall
    be extended by one semester even with a single backlog course and you will be
    required to pay the full semester fee for that semester registration.
  - Students are allowed to register for a maximum of four courses in one semester.
    If you have any pending backlog courses, the duration of your programme shall
    be extended by one semester even with a single backlog course and you will be
    required to pay the full semester fee for that semester registration.
  - BITS Pilani does not issue certificates regarding the intellectual property rights
    or copyright of student theses, research, or projects. ● If your organization
    restricts the use of proprietary IP, consider working with open-source data or
    projects that do not involve confidential information. ● For BITS evaluation,
    a demo of your project—including code and data—is mandatory. You may use relevant
    or open-source data for this purpose. ● If your code is confidential, it does
    not need to be included in the report. ● For projects on open-source work outside
    your organization, obtain a No Objection Certificate (NOC) in any suitable format
    from your employer and share it with support@wilp.bits-pilani.ac.in.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'How do I register for my previous semester backlog course(s)?',
    'Students are allowed to register for a maximum of four courses in one semester. If you have any pending backlog courses, the duration of your programme shall be extended by one semester even with a single backlog course and you will be required to pay the full semester fee for that semester registration.',
    'Students are allowed to register for a maximum of four courses in one semester. If you have any pending backlog courses, the duration of your programme shall be extended by one semester even with a single backlog course and you will be required to pay the full semester fee for that semester registration.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6177, 0.6177],
#         [0.6177, 1.0000, 1.0000],
#         [0.6177, 1.0000, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 344 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 344 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         | sentence_2                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                              |
  | details | <ul><li>min: 5 tokens</li><li>mean: 16.51 tokens</li><li>max: 40 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 69.3 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 67.31 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                    | sentence_1                                                                                                                                                                                                                                                                                                                     | sentence_2                                                                                                                                                                                                                                                                                                                     |
  |:------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>How do I register for my previous semester backlog course(s)?</code>    | <code>Students are allowed to register for a maximum of four courses in one semester. If you have any pending backlog courses, the duration of your programme shall be extended by one semester even with a single backlog course and you will be required to pay the full semester fee for that semester registration.</code> | <code>Students are allowed to register for a maximum of four courses in one semester. If you have any pending backlog courses, the duration of your programme shall be extended by one semester even with a single backlog course and you will be required to pay the full semester fee for that semester registration.</code> |
  | <code>What is the eligibility to be a supervisor?</code>                      | <code>A Supervisor should possess one of the following qualifications: B.E., B.Tech., M.Sc., MBA, or MCA, along with a minimum of 5 years of relevant work experience.</code>                                                                                                                                                  | <code>A Supervisor should possess one of the following qualifications: B.E., B.Tech., M.Sc., MBA, or MCA, along with a minimum of 5 years of relevant work experience.</code>                                                                                                                                                  |
  | <code>What should I do if my TURNITIN account gets deleted by mistake?</code> | <code>Please contact support at support@wilp.bits-pilani.ac.in for assistance.</code>                                                                                                                                                                                                                                          | <code>Please contact support at support@wilp.bits-pilani.ac.in for assistance.</code>                                                                                                                                                                                                                                          |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.COSINE",
      "triplet_margin": 5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.11.11
- Sentence Transformers: 5.2.0
- Transformers: 4.57.6
- PyTorch: 2.9.1+cu128
- Accelerate: 1.12.0
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->