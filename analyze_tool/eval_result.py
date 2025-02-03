import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description="Run SuperGLUE inference and evaluation.")
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="The dataset class name to use, e.g., 'SuperGLUEDataset'."
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="The model wrapper class name to use, e.g., 'QWen2VLWrapper'."
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['text', 'img', 'semi'],
        help="The mode for inference and evaluation. Options are 'text', 'img', or 'semi'."
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        choices=['base', 'cot'],
        help="The prompt type for text data. Options are 'base' or 'cot'."
    )
    parser.add_argument(
        '--vector',
        action='store_true',
        help="Enable or disable vector mode."
    )

    args = parser.parse_args()
    from model import load_model_input_name
    model_input_name = load_model_input_name(args.model)
    
    try:
        data_module = importlib.import_module('data')
        dataset_class = getattr(data_module, args.dataset)
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Model wrapper class '{args.dataset}' not found in 'data' module.")

    dataset = dataset_class(cot_flag = args.prompt == 'cot', mode=args.mode, init_data=False)
    if args.vector:
        dataset.cache_path = "./result_cache_vector/result_cache/"
    dataset.Eval(model_input_name)

if __name__ == '__main__':
    main()