import json
import importlib
import signal
from contextlib import contextmanager

def timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out.")

@contextmanager
def time_limit(seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def getEvalResult(model, data, mode, prompt, eval_cache="eval_cache.json", error_cache="error_cache.json", errorInfo_path="remend_exp.sh"):
    try:
        # Generate cache key
        cache_key = f"{model}_{data}_{mode}_{prompt}"

        # Load evaluation cache
        try:
            with open(eval_cache, "r") as cache_file:
                cache = json.load(cache_file)
        except FileNotFoundError:
            cache = {}

        # Load error cache
        try:
            with open(error_cache, "r") as error_file:
                error_cache_data = json.load(error_file)
        except FileNotFoundError:
            error_cache_data = {}

        # Check if the configuration is in the error cache
        if cache_key in error_cache_data:
            print("Error cache hit: Skipping evaluation.")
            return None

        # Check if result is already cached
        if cache_key in cache:
            print("Cache hit: Returning cached results.")
            return cache[cache_key]["task_results"], cache[cache_key]["final_score"]

        from model import load_model_input_name
        model_name = load_model_input_name(model)

        data_module = importlib.import_module('data')
        dataset_class = getattr(data_module, data)

        try:
            with time_limit(600):
                # Create dataset instance and evaluate with a timeout
                dataset = dataset_class(cot_flag=(prompt == 'cot'), mode=mode, init_data=False)
                try:
                    task_results, final_score = dataset.Eval(model_name)
                except Exception as e:
                    dataset.cache_path = "./result_cache_vector/result_cache/"
                    task_results, final_score = dataset.Eval(model_name)
        except TimeoutError:
            print("Evaluation timed out.")
            raise TimeoutError("Evaluation timed out.")

        # Cache the evaluation results
        cache[cache_key] = {
            "model": model,
            "dataset": data,
            "mode": mode,
            "prompt": prompt,
            "task_results": task_results,
            "final_score": final_score
        }

        with open(eval_cache, "w") as cache_file:
            json.dump(cache, cache_file, indent=4)

        return task_results, final_score

    except Exception as e:
        if str(e) == "list index out of range":
            raise e
        # Log the error configuration in error cache
        error_cache_data[cache_key] = {
            "model": model,
            "dataset": data,
            "mode": mode,
            "prompt": prompt,
            "error": str(e)
        }

        with open(error_cache, "w") as error_file:
            json.dump(error_cache_data, error_file, indent=4)

        # Write the error command to the errorInfo_path file
        command = f"nohup python data.py --dataset {data} --model {model} --mode {mode} --prompt {prompt} > {data}_{model}_{mode}_{prompt}.log 2>&1 &\n"
        with open(errorInfo_path, "a") as error_file:
            error_file.write(command)

        print(f"Error occurred: {str(e)}")
        print(f"Configuration logged in error cache: {model}, {data}, {mode}, {prompt}")
        return None

def checkAllExperiment(reCheck=False):
    model_list = ["Gemini_Flash", "Gemini_Thinking", "GPT4o", "Qwen2_VL_72B", "Qwen2_VL_7B", "Qwen2_VL_2B",
                  "QVQ_72B_Preview", "InternVL2-Llama3-76B", "InternVL2-8B",
                  # "Llava_OneVision_72B", "Llava_OneVision_7B",
                  "Pixtral-12B", "Phi_3_5_vision"]
    data_list = ["SuperGLUEDataset", "GLUEDataset", "MMLUDataset", "MMLUProDataset",
                 "GSM8KDataset", "MBPPDataset", "ARCDataset", "TableBenchDataset",
    "WikiSS_QADataset", "SlidesVQADataset", "MathverseDataset"]

    # model_list = ["Qwen2_VL_7B"]
    # data_list = ["SuperGLUEDataset"]

    # mode option: text, img; semi if TableBench
    # prompt option: base, cot

    # clean the errorInfo_path file
    if reCheck: # clean all three files
        with open("eval_cache.json", "w") as cache_file:
            cache_file.write("{}")
        with open("error_cache.json", "w") as error_file:
            error_file.write("{}")
        with open("remend_exp.sh", "w") as error_file:
            error_file.write("")
    # else: # clean the errorInfo_path file
    #     with open("remend_exp.sh", "w") as error_file:
    #         error_file.write("")

    for model in model_list:
        for data in data_list:
            for prompt in ["base", "cot"]:
                for mode in ["text", "img"]:
                    getEvalResult(model, data, mode, prompt, "eval_cache.json", "error_cache.json", "remend_exp.sh")
                if data == "TableBenchDataset":
                    getEvalResult(model, data, "semi", prompt, "eval_cache.json", "error_cache.json", "remend_exp.sh")

if __name__ == "__main__":
    checkAllExperiment(reCheck=True)
    # getEvalResult("Qwen2_VL_7B", "SuperGLUEDataset", "img", "base", "eval_cache.json", "error_cache.json",
    #               "remend_exp.sh")