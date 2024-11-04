def default_parse(parser):

    # (1) General Settings

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bf16", action='store_true')
    parser.add_argument("--fp32", action='store_true')
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--tf32", action='store_true')
    parser.add_argument("--api", action='store_true')
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument("--cache_dir",default='/projects/minsu')


 
    # 2. Dataset
    parser.add_argument("--data_name", default="holism_test", type=str,
                        help='')
    parser.add_argument("--test_set", default="/userhomes/minsu/EviConf/data/SciQ/Final/SciQ_original.jsonl")


    # 3. Model
    parser.add_argument("--load_pretrained", default=True, type=bool,
                        help='Bring pre-trained FLAN-T5 model.')
    parser.add_argument("--model_name", default="google/flan-t5-large", type=str,
                        help='Baseline FLAN-T5 model for self-tuning.')
    parser.add_argument("--output_min_length", default=1, type=int)
    parser.add_argument("--output_max_length", default=128, type=int)
    parser.add_argument("--response_model", default='llama-7b', type=str)
    parser.add_argument("--answer_strategy", default='evidence', type=str, help="evidence,direct")
    parser.add_argument("--confidence_method", default='verbal',type=str, help="verbal, token_prob, sampling")
    parser.add_argument("--num_sample", default=10, type=int, help="5 or 10")
    # Wandb
    parser.add_argument("--project", default="EviConf", type=str)
    parser.add_argument("--entity", default="kimminsu")
    parser.add_argument("--run_name", default="RL", type=str)

    args = parser.parse_args()

    return args