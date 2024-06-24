import wandb
from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers
import smplx_utils as smpllm
import json
import re

NO_BETAS_FOUND_PENALTY = 2
W_DEFAULT_LOSS = 0.5
W_BETAS_LOSS = 1.5
W_MEASUREMENTS_LOSS = 2.5
BETAS_WEIGHT = [2.4938, 0.8701, 0.6369, 0.4250, 0.3144, 0.2967, 0.2287, 0.1530, 0.1456, 0.1268]

with open('/home/baldo/PycharmProjects/qloraTraining/smplx-code/data/smpllm_ranges_v1_0.json',
          'r') as ranges_file:
    ranges_data = json.load(ranges_file)

ranges_list = list(ranges_data.values())
ranges_tensor = torch.tensor(ranges_list, device='cuda')

betas_weight_tensor = torch.tensor(BETAS_WEIGHT, device='cuda')

print("CUDA AVAILABLE: " + str(torch.cuda.is_available()))

# Step 1 - CARGAR MODELO BASE ------------------------------------------------------------------------------------------------
print("[DEB] Starting step 1: Load Dataset")

train_dataset = load_dataset('json', data_files='training_desc.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='validation_desc.jsonl', split='train')


def formatting_func(example):
    text = f"### Description: {example['input']}\n ### Shape parameters: {example['output']}"
    return text


# Step 2 - CARGAR MODELO BASE ---------------------------------------------------------------------------------------------
print("[DEB] Starting step 2: Load Base Model")

#base_model_id = "meta-llama/Llama-2-7b-hf"
base_model_id = "meta-llama/Meta-Llama-3-8B"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="cuda")

# Step 3 - TOKENIZACION ------------------------------------------------------------------------------------------------
print("[DEB] Starting step 3: Tokenization")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

max_length = 256

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)

# Step 4 - CONFIGURAR LORA -------------------------------------------------------------------------------------------------
print("[DEB] Starting step 4: Set Up LoRA")

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

# Step 4.1 - ERROR PERSONALIZADO ----------------------------------------------------------------------------------------------
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )

        # Extraer labels y target text
        labels = inputs["labels"]
        labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
        target_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Extraer logits y predicted text
        logits = outputs["logits"]
        predicted_text = tokenizer.batch_decode(logits.argmax(-1), skip_special_tokens=True)

        # Extraer las betas del texto. Si alguna beta no es encontrada (hay 9 o menos betas), se coloca un 0 y se
        # penaliza en el error final.
        incomplete_betas = check_betas(target_text, 10)  # Incomplete_betas: número de textos con menos de 10 betas.
        # Idealmente debería ser 0. Se penalizará en el cálculo del error.
        target_betas = extract_betas(target_text)
        predicted_betas = extract_betas(predicted_text)

        # Default loss computed from base code
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Calcular el error en la diferencia de las betas
        betas_loss = betas_difference(target_betas, predicted_betas, betas_weight_tensor)

        # Calcular measurements
        target_measurements = betas_to_measurements(target_betas)
        predicted_measurements = betas_to_measurements(predicted_betas)

        # Calcular las categorías de las medidas en función de los rangos
        target_categories = measurements_to_categories(target_measurements, ranges_tensor)
        predicted_categories = measurements_to_categories(predicted_measurements, ranges_tensor)
        predicted_categories_one_hot = F.one_hot(predicted_categories, num_classes=5)
        predicted_categories_one_hot = predicted_categories_one_hot.float()
        measurements_loss = F.cross_entropy(predicted_categories_one_hot.permute(0, 2, 1), target_categories)

        default_loss_value = loss.item()
        betas_loss_value = betas_loss.item()
        measurements_loss_value = measurements_loss.item()
        # Mezclar error
        merged_loss = (W_DEFAULT_LOSS * default_loss_value +
                       W_BETAS_LOSS * betas_loss_value +
                       W_MEASUREMENTS_LOSS * measurements_loss_value +
                       NO_BETAS_FOUND_PENALTY * incomplete_betas)
        # Aplicar error
        loss.data = torch.tensor(merged_loss, device='cuda:0')

        # Wandb log
        wandb.log({"default loss": default_loss_value})
        wandb.log({"betas loss": betas_loss_value})
        wandb.log({"measurements loss": measurements_loss_value})
        wandb.log({"incomplete betas penalty": (incomplete_betas * NO_BETAS_FOUND_PENALTY)})

        if predicted_betas.size(0) == 10: # Paso de evaluación
            wandb.log({"default loss_EVAL": default_loss_value})
            wandb.log({"betas loss_EVAL": betas_loss_value})
            wandb.log({"measurements loss_EVAL": measurements_loss_value})
            wandb.log({"incomplete betas penalty_EVAL": (incomplete_betas * NO_BETAS_FOUND_PENALTY)})

        # Imprimir todos los valores
        print("DEFAULT_LOSS: " + str(default_loss_value) +
              "\nBETAS LOSS : " + str(betas_loss_value) +
              "\nMEASUREMENTS LOSS : " + str(measurements_loss_value) +
              "\nTARGET TEXT: " + str(target_text) +
              "\nPREDICTION TEXT: " + str(predicted_text))

        return (loss, outputs) if return_outputs else loss


def convert_labels_to_text(labels):
    label_texts = []
    for label in labels:
        label_text = tokenizer.decode(label.item(), skip_special_tokens=True)
        label_texts.append(label_text)


def check_betas(text_list,
                num):  # Comprueba cuántos elementos de la lista de strings predicha no tienen 10 decimales para penalizar posteriormente.
    count = 0
    for text in text_list:
        decimal_numbers = re.findall(r'-?\d+\.\d+', text)
        if len(decimal_numbers) < num:
            count += 1
    return count


def extract_betas(
        text_list):  # Extrae las 10 primeras betas de cada string de la lista, y completa con ceros si falta alguna.
    tensor_list = []
    for text in text_list:
        decimal_numbers = re.findall(r'-?\d+\.\d+', text)
        decimal_numbers = [float(num) for num in decimal_numbers]
        if len(decimal_numbers) < 10:
            decimal_numbers += [0.0] * (10 - len(decimal_numbers))
        tensor_list.append(decimal_numbers[:10])

    return torch.tensor(tensor_list, device='cuda')


def custom_loss_betas(predicted, target):
    predicted_betas = extract_betas(predicted)
    target_betas = extract_betas(target)
    weights = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    loss = 0
    for i in range(len(predicted_betas)):
        loss += abs(predicted_betas[i] - target_betas[i]) * weights[i]
    loss /= len(predicted_betas)
    return loss


def custom_loss_cross_entropy(predicted, target, ranges):
    predicted_betas = extract_betas(predicted)
    target_betas = extract_betas(target)

    predicted_tensor = torch.tensor([predicted_betas], device='cuda', dtype=torch.float32)
    target_tensor = torch.tensor([target_betas], device='cuda', dtype=torch.float32)

    predicted_eval = smpllm.get_evaluation_vector_expanded(predicted_tensor, ranges)
    target_eval = smpllm.get_evaluation_vector(target_tensor, ranges)

    loss = F.cross_entropy(predicted_eval, target_eval.squeeze().long())
    return loss


def betas_difference(target_betas, predicted_betas, weights):
    abs_diff = torch.abs(target_betas - predicted_betas)
    weighted_diff = abs_diff * weights
    sum_weighted_diff = torch.sum(weighted_diff, dim=1)
    num_elements = target_betas.size(1)
    mean_weighted_diff = sum_weighted_diff / num_elements
    global_mean_weighted_diff = torch.mean(mean_weighted_diff)
    return global_mean_weighted_diff.unsqueeze(0)


def betas_to_measurements(betas):
    avatar_model, output, vertices = smpllm.generate_avatar_from_betas_batch(betas)

    measurements = smpllm.get_measurements_batch(avatar_model, vertices)

    return measurements


def measurements_to_categories(measurements, ranges_input):
    num_subjects, num_measures = measurements.size()
    num_ranges = ranges_input.size(1)

    labeled_measures = torch.zeros(num_subjects, num_measures, dtype=torch.long, device='cuda')

    for i in range(num_measures):
        measure = measurements[:, i].unsqueeze(1)  # Convertir a columna
        ranges = ranges_input[i]

        # Comparar la medida con los rangos y asignar etiqueta
        labels = torch.sum(measure >= ranges, dim=1)
        labeled_measures[:, i] = labels

    return labeled_measures


# Step 5 - LANZAR ENTRENAMIENTO ------------------------------------------------------------------------------------------------
print("[DEB] Starting step 5: Run Training")
if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

project = "smpllm-finetune-v24"
base_model_name = "llama3-8b"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

# Configurar entrenamiento
trainer = CustomTrainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        use_cpu=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=50000,
        learning_rate=1e-6,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=2000,
        evaluation_strategy="steps",
        eval_steps=1000,
        do_eval=True,
        do_train=True,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False
trainer.train()
