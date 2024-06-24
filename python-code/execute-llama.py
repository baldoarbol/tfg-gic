import random

from openai import OpenAI

client = OpenAI(base_url="http://localhost:7654/v1", api_key="not-needed")

import jsonlines


def rewording(input_desc, header):
    completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system",
             "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
            {"role": "user",
             "content": header + input_desc}
        ],
        temperature=0.7,
    )
    rewording_output_raw = completion.choices[0].message
    rewording_output = rewording_output_raw.content
    answer = {"rewording": rewording_output}
    return answer


# Archivo de entrada y salida
archivo_entrada = "/home/baldo/PycharmProjects/qloraTraining/smplx-code/data/validation_desc.jsonl"
archivo_salida = "/home/baldo/PycharmProjects/qloraTraining/smplx-code/data/validation_desc_rewording0.jsonl"

headerA = 'I will send you a body description. Rewrite this description using different words. Provide ONLY the description and nothing else: '
headerB = 'I will send you a body description. Rewrite this description but change the order of the attributes. Provide ONLY the description and nothing else: '

# Abrir archivo de entrada y leer líneas
with jsonlines.open(archivo_entrada) as reader:
    # Abrir archivo de salida para escritura
    with jsonlines.open(archivo_salida, mode='w') as writer:
        # Procesar cada línea del archivo de entrada
        n = 0
        for entrada in reader:
            random_num = random.randint(1, 3)
            if random_num == 1:
                respuesta = {"rewording": entrada["input"]}
            elif random_num == 2:
                respuesta = rewording(entrada["input"], headerA)
            elif random_num == 3:
                respuesta = rewording(entrada["input"], headerB)
            n += 1
            print("STEP " + str(n) + "/2000" + ": [" + str(random_num) + "]" + respuesta["rewording"])
            # Escribir el contenido en el archivo de salida
            writer.write(respuesta)