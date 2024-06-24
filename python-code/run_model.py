import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import json
from peft import PeftModel
import smplx_utils as smpllm
import matplotlib.pyplot as plt

MODEL_ID_DEFAULT = "llama3-8b-smpllm-finetune-v21/checkpoint-10000"
load_model = True
DATASET_CONTROL_FILE = "/home/baldo/PycharmProjects/qloraTraining/smplx-code/data/control_desc_rewording1.jsonl"
RANGES_FILE = '/home/baldo/PycharmProjects/qloraTraining/smplx-code/data/smpllm_ranges_v1_0.json'

MAIN_EXPORT_FILE_PATH = '/home/baldo/PycharmProjects/qloraTraining/control/'
CONTROL_FILE_NAME = "evaluation_control_B03.json"
REPORT_FILE_NAME = "report_control_B03.json"
GRAPHS_FOLDER_NAME = "graphs_control_B03/"


def main():
    test_data = run_test_comparative_numbers(1000)
    expected_betas_tensor = torch.tensor(test_data["expected betas"], device='cuda')
    generated_betas_tensor = torch.tensor(test_data["generated betas"], device='cuda')

    expected_measures_tensor = betas_to_measurements(expected_betas_tensor)
    generated_measures_tensor = betas_to_measurements(generated_betas_tensor)

    with open(RANGES_FILE,'r') as ranges_file:
        ranges_data = json.load(ranges_file)

    evaluate_generated_over_ranges(expected_measures_tensor, generated_measures_tensor, ranges_data)



def betas_to_measurements(betas):
    avatar_model, output, vertices = smpllm.generate_avatar_from_betas_batch(betas)
    measurements = smpllm.get_measurements_batch(avatar_model, vertices)
    return measurements


if load_model:
    #base_model_id = "meta-llama/Llama-2-7b-hf"
    base_model_id = "meta-llama/Meta-Llama-3-8B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

    ft_model = PeftModel.from_pretrained(base_model, MODEL_ID_DEFAULT)


def extract_betas(text):
    # Buscar todos los números decimales del texto usando regex
    decimal_numbers = re.findall(r'-?\d+\.\d+', text)
    decimal_numbers = [float(num) for num in decimal_numbers]
    # Devolver los 10 primeros números
    return decimal_numbers[:10]


def read_jsonl(archivo_jsonl):
    with open(archivo_jsonl, 'r') as file:
        content = [json.loads(line.strip()) for line in file]
    return content


def run_test_comparative_numbers(num):
    test_data = read_jsonl(DATASET_CONTROL_FILE)

    descriptions_global = []
    expected_betas_global = []
    generated_betas_global = []
    local_split_difs = []
    local_avg_difs = []
    total_avg_dif = 0

    for i in range(num):
        description = test_data[i]["input"]
        ground_truth = extract_betas(test_data[i]["output"])
        answer_text = run_model(description)
        generated_betas = extract_betas(answer_text)

        split_difs = [0] * 10
        local_dif_avg = 0

        for j in range(10):
            if len(generated_betas) < 10:
                generated_index = 0
            else:
                generated_index = generated_betas[j]
            split_difs[j] = generated_index - ground_truth[j]
            local_dif_avg += abs(split_difs[j])

        local_dif_avg /= 10

        descriptions_global.append(description)
        expected_betas_global.append(ground_truth)
        generated_betas_global.append(generated_betas)
        local_split_difs.append(split_difs)
        local_avg_difs.append(local_dif_avg)
        total_avg_dif += local_dif_avg

        print("Progreso: " + str(i + 1) + "/" + str(num))

    total_avg_dif = total_avg_dif / num

    test_data = {
        "total error average": total_avg_dif,
        "descriptions": descriptions_global,
        "expected betas": expected_betas_global,
        "generated betas": generated_betas_global,
        "split errors": local_split_difs,
        "avg errors": local_avg_difs
    }

    export_file_path = MAIN_EXPORT_FILE_PATH + CONTROL_FILE_NAME

    if not os.path.exists(export_file_path):
        os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
        open(export_file_path, "w").close()

    with open(export_file_path, "w") as file:
        json.dump(test_data, file)

    print("Datos escritos en disco en el archivo: " + export_file_path)

    return test_data


def run_model(input_text):
    description = input_text
    eval_prompt = "### Description: " + description + "\n ### Shape parameters: "
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    ft_model.eval()
    with torch.no_grad():
        answer = tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=200)[0], skip_special_tokens=True)
    return answer


def evaluate_generated_over_ranges(expected_measures_tensor, generated_measures_tensor, ranges_data):
    measurements_data = {
        "measurements_expected": {},
        "measurements_generated": {}
    }
    expected_measures_list = expected_measures_tensor.transpose(0,1).tolist()
    generated_measures_list = generated_measures_tensor.transpose(0,1).tolist()
    index = 0
    for measurement_ in smpllm.measurements_to_export:
        measurements_data["measurements_expected"][measurement_.value] = expected_measures_list[index]
        measurements_data["measurements_generated"][measurement_.value] = generated_measures_list[index]
        index += 1

    # Inicializar la lista de medidas
    measurements_list = []
    result_analysis = {}
    ranges_key = ["_very_low", "_low", "_average", "_high", "_very_high", "_total"]

    for key_ in ranges_data.keys():
        measurements_list.append(key_)
        result_analysis[key_ + ranges_key[5]] = (-1, -1)
        for i in range(5):
            result_analysis[key_ + ranges_key[i]] = (-1, -1)

    # Loop principal de cada medida
    for measurement_ in measurements_list:
        # Variables para la gráfica
        ranges_limits = ranges_data[measurement_]
        ranges = []
        generated_list = measurements_data["measurements_generated"][measurement_]
        expected_list = measurements_data["measurements_expected"][measurement_]
        expected_ranges = []
        min_value = min(measurements_data["measurements_generated"][measurement_])
        max_value = max(measurements_data["measurements_generated"][measurement_])
        # Se añaden el límite superior y límite inferior, que va desde el límite establecido hasta el último vañor o crea un rango tan grande como el rango anterior
        ranges_limits.append(max(max_value, ranges_limits[3] + (ranges_limits[3] - ranges_limits[2])))
        ranges_limits.insert(0, min(min_value, ranges_limits[0] - (ranges_limits[1] - ranges_limits[0])))

        # Variables para el informe
        total_accuracy = 0
        average_distance = 0
        local_accuracies = [0] * 5
        local_distances = [0] * 5
        num_instances_per_range = [0] * 5

        # Generar rangos
        for i in range(5):
            range_ = (ranges_limits[i], ranges_limits[i + 1])
            ranges.append(range_)

        # Asignar a cada valor el rango esperado
        for i in range(len(expected_list)):
            expected_range_ = 0
            for j in range(5):
                if ranges_limits[j] <= expected_list[i] < ranges_limits[j + 1]:
                    expected_range_ = j
            expected_ranges.append(expected_range_)

        # Elaboración del report
        for i in range(len(generated_list)):
            expected_r = expected_ranges[i]
            num_instances_per_range[expected_r] += 1
            top = ranges_limits[expected_r + 1]
            bottom = ranges_limits[expected_r]
            mid = top - bottom / 2
            if bottom <= generated_list[i] < top:
                local_accuracies[expected_r] += 1
                total_accuracy += 1
            distance = abs(expected_r - generated_list[i])
            local_distances[expected_r] += distance
            average_distance += distance
        total_accuracy /= len(generated_list)
        average_distance /= len(generated_list)
        for i in range(len(local_accuracies)):
            local_accuracies[i] /= max(1, num_instances_per_range[i])
            local_distances[i] /= max(1, num_instances_per_range[i])

        result_analysis[measurement_ + ranges_key[5]] = (total_accuracy, average_distance)
        for i in range(5):
            result_analysis[measurement_ + ranges_key[i]] = (local_accuracies[i], local_distances[i])

        datos = generated_list
        rangos = ranges
        rangos_deseados = expected_ranges

        # Dibuja los rangos en el gráfico principal
        for i, rango in enumerate(rangos):
            plt.plot(rango, [i, i], color='C{}'.format(i), linewidth=10, label='_nolegend_', zorder=1,
                     solid_capstyle='butt')

        # Dibuja los rangos en la leyenda con línea fina
        labels = ["Valor muy bajo", "Valor bajo", "Valor promedio", "Valor alto", "Valor muy alto"]
        handles = []
        for i, rango in enumerate(rangos):
            line = \
            plt.plot([], [], color='C{}'.format(i), linewidth=4, label=labels[i], zorder=1, solid_capstyle='butt')[0]
            handles.insert(0, line)  # Insertar al inicio de la lista para invertir el orden

        # Dibuja los puntos
        for dato, rango_deseado in zip(datos, rangos_deseados):
            plt.scatter(dato, rango_deseado, color='C{}'.format(rango_deseado), marker='o', s=15, edgecolor='black',
                        linewidth=0.4, zorder=2)

        plt.xlabel('Valor')
        plt.ylabel('Rango deseado')
        plt.title('Distribución de datos en rangos deseados: ' + measurement_)

        # Ajusta la leyenda
        plt.legend(handles=handles, title='Rangos')
        plt.grid(True)
        file_output = MAIN_EXPORT_FILE_PATH + GRAPHS_FOLDER_NAME + "graph_" + measurement_
        plt.savefig(file_output)
        plt.show()

    report_file_path = MAIN_EXPORT_FILE_PATH + REPORT_FILE_NAME

    if not os.path.exists(report_file_path):
        os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
        open(report_file_path, "w").close()

    with open(report_file_path, "w") as json_file:
        json.dump(result_analysis, json_file, indent=2)

    print(f"Informe guardado en el archivo: {report_file_path}")


if __name__ == "__main__":
    main()
