import trimesh
import numpy as np
import torch
import smplx
from enum import Enum
import random


# Clase que almacena el ID de todos los vértices clave.
class KeyVertex(Enum):
    HEAD = 9011
    LEFT_HIP = 5504
    LEFT_SHOULDER = 4482
    NECK = 5518
    RIGHT_ANKLE = 6438
    RIGHT_FOOT = 8651
    RIGHT_HIP = 8226
    RIGHT_SHOULDER = 6626
    RIGHT_WRIST = 7559


# Tipos de medidas
class MeasurementType(Enum):
    VERTEX_DISTANCE = 0
    THICKNESS = 1
    VOLUME = 2
    RELATION = 3


# Clase que se utiliza siempre que se hace referencia a una medida
class Measurement(Enum):
    HEIGHT = "height"  # Final
    LEGS_LENGTH = "legs_length"
    ARMS_LENGTH = "arms_length"
    SHOULDERS_DISTANCE = "shoulders_distance"
    HIP_DISTANCE = "hip_distance"
    NECK_LENGTH = "neck_length"  # Final
    BMI = "bmi"  # Final
    LEGS_RELATION = "legs_relation"  # Final
    ARMS_RELATION = "arms_relation"  # Final
    SHOULDERS_RELATION = "shoulders_relation"  # Final
    HIP_RELATION = "hip_relation"  # Final
    ARM_THICKNESS = "arm_thickness"  # Final
    LEG_THICKNESS = "leg_thickness"  # Final
    BUST_THICKNESS = "bust_thickness"  # Final
    WAIST_THICKNESS = "waist_thickness"  # Final
    HIP_THICKNESS = "hip_thickness"  # Final
    VOLUME = "volume"


"""
Global variables
"""

MODEL_FOLDER = '/home/baldo/PycharmProjects/qloraTraining/smplx-code/smplx/models'
MODEL_TYPE = 'smplx'
GENDER = "neutral"
NUM_BETAS = 10

# Lista de medidas "fnales", medidas que se evalúan como atributos del avatar.
measurements_to_export = [
    Measurement.HEIGHT,
    Measurement.NECK_LENGTH,
    Measurement.LEGS_RELATION,
    Measurement.ARMS_RELATION,
    Measurement.SHOULDERS_RELATION,
    Measurement.HIP_RELATION,
    Measurement.BMI,
    Measurement.ARM_THICKNESS,
    Measurement.LEG_THICKNESS,
    Measurement.BUST_THICKNESS,
    Measurement.WAIST_THICKNESS,
]

# Vértices implicados en las medidas de tipo "vertex distance"
vertex_distance_dictionary = {
    Measurement.HEIGHT: [KeyVertex.RIGHT_FOOT, KeyVertex.HEAD],
    Measurement.ARMS_LENGTH: [KeyVertex.RIGHT_SHOULDER, KeyVertex.RIGHT_WRIST],
    Measurement.LEGS_LENGTH: [KeyVertex.RIGHT_HIP, KeyVertex.RIGHT_ANKLE],
    Measurement.SHOULDERS_DISTANCE: [KeyVertex.RIGHT_SHOULDER, KeyVertex.LEFT_SHOULDER],
    Measurement.NECK_LENGTH: [KeyVertex.NECK, KeyVertex.HEAD],
    Measurement.HIP_DISTANCE: [KeyVertex.RIGHT_HIP, KeyVertex.LEFT_HIP],
}

# Vértices implicados en las medidas de tipo "thickness"
vertex_thickness_dictionary = {
    # Vértice más alto / más bajo / más adelantado / más atrasado
    Measurement.ARM_THICKNESS: [6699, 6761, 6022, 6767],
    # Vértice más adelantado / más atrasado / más a la derecha del modelo / más a la izquierda del modelo
    Measurement.LEG_THICKNESS: [6266, 6291, 6296, 6354],
    Measurement.BUST_THICKNESS: [5938, 5499, 6131, 3925],
    Measurement.WAIST_THICKNESS: [4292, 4298, 8225, 5504],
    Measurement.HIP_THICKNESS: [5615, 5934, 6203, 3439]
}

# Medidas implicadas en cada relación de medidas
measurement_relations_dictionary = {
    Measurement.BMI: [Measurement.VOLUME, Measurement.HEIGHT],
    Measurement.LEGS_RELATION: [Measurement.LEGS_LENGTH, Measurement.HEIGHT],
    Measurement.ARMS_RELATION: [Measurement.ARMS_LENGTH, Measurement.HEIGHT],
    Measurement.SHOULDERS_RELATION: [Measurement.SHOULDERS_DISTANCE, Measurement.HEIGHT],
    Measurement.HIP_RELATION: [Measurement.HIP_DISTANCE, Measurement.HEIGHT]
}

# Tipo de cada medida
measurement_types_dictionary = {
    Measurement.HEIGHT: MeasurementType.VERTEX_DISTANCE,
    Measurement.LEGS_LENGTH: MeasurementType.VERTEX_DISTANCE,
    Measurement.ARMS_LENGTH: MeasurementType.VERTEX_DISTANCE,
    Measurement.SHOULDERS_DISTANCE: MeasurementType.VERTEX_DISTANCE,
    Measurement.HIP_DISTANCE: MeasurementType.VERTEX_DISTANCE,
    Measurement.NECK_LENGTH: MeasurementType.VERTEX_DISTANCE,
    Measurement.VOLUME: MeasurementType.VOLUME,
    Measurement.BMI: MeasurementType.RELATION,
    Measurement.LEGS_RELATION: MeasurementType.RELATION,
    Measurement.ARMS_RELATION: MeasurementType.RELATION,
    Measurement.SHOULDERS_RELATION: MeasurementType.RELATION,
    Measurement.HIP_RELATION: MeasurementType.RELATION,
    Measurement.ARM_THICKNESS: MeasurementType.THICKNESS,
    Measurement.LEG_THICKNESS: MeasurementType.THICKNESS,
    Measurement.BUST_THICKNESS: MeasurementType.THICKNESS,
    Measurement.WAIST_THICKNESS: MeasurementType.THICKNESS,
    Measurement.HIP_THICKNESS: MeasurementType.THICKNESS
}

# Diccionario de expresiones verbales para depuración
evaluations_dictionary = {
    Measurement.HEIGHT: ['MUY BAJO', 'BAJO', 'altura promedio', 'ALTO', 'MUY ALTO'],
    Measurement.NECK_LENGTH: ['CUELLO MUY CORTO', 'CUELLO CORTO', 'cuello promedio', 'CUELLO LARGO',
                              'CUELLO MUY LARGO'],
    Measurement.BMI: ['MUY POCO CORPULENTO', 'POCO CORPULENTO', 'talla promedio', 'CORPULENTO', 'MUY CORPULENTO'],
    Measurement.LEGS_RELATION: ['PIERNAS MUY CORTAS', 'PIERNAS CORTAS', 'piernas promedio', 'PIERNAS LARGAS',
                                'PIERNAS MUY LARGAS'],
    Measurement.ARMS_RELATION: ['BRAZOS MUY CORTOS', 'BRAZOS CORTOS', 'brazos promedio', 'BRAZOS LARGOS',
                                'BRAZOS MUY LARGOS'],
    Measurement.SHOULDERS_RELATION: ['MUY ESTRECHO DE HOMBROS', 'ESTRECHO DE HOMBROS', 'anchura de hombros promedio',
                                     'ANCHO DE HOMBROS', 'MUY ANCHO DE HOMBROS'],
    Measurement.HIP_RELATION: ['MUY ESTRECHO DE CADERA', 'ESTRECHO DE CADERA', 'anchura de cadera promedio',
                               'ANCHO DE CADERA', 'MUY ANCHO DE CADERA'],
    Measurement.ARM_THICKNESS: ['BRAZOS MUY DELGADOS', 'BRAZOS DELGADOS', 'grosor de brazos promedio', 'BRAZOS GRUESOS',
                                'BRAZOS MUY GRUESOS'],
    Measurement.LEG_THICKNESS: ['PIERNAS MUY DELGADAS', 'PIERNAS DELGADAS', 'grosor de piernas promedio',
                                'PIERNAS GRUESAS', 'PIERNAS MUY GRUESAS'],
    Measurement.BUST_THICKNESS: ['PECHO MUY DELGADO', 'PECHO DELGADO', 'grosor de pecho promedio',
                                 'PECHO GRUESO', 'PECHO MUY GRUESO'],
    Measurement.HIP_THICKNESS: ['CADERAS MUY DELGADAS', 'CADERAS DELGADAS', 'grosor de caderas promedio',
                                'CADERAS GRUESAS', 'CADERAS MUY GRUESAS'],
    Measurement.WAIST_THICKNESS: ['CINTURA MUY DELGADA', 'CINTURA DELGADA', 'grosor de cintura promedio',
                                  'CINTURA GRUESA', 'CINTURA MUY GRUESA']
}

# Diccionario de expresiones verbales para descripciones completas
textual_expressions_english = {
    Measurement.HEIGHT: ['very short', 'short', 'average height', 'tall', 'very tall'],
    Measurement.NECK_LENGTH: ['a very short neck', 'a short neck', 'neck of average length',
                              'tall neck',
                              'very tall neck'],
    Measurement.BMI: ['very little body mass', 'low body mass', 'average body mass',
                      'a lot of body mass', 'very much body mass'],
    Measurement.LEGS_RELATION: ['very short legs', 'short legs', 'legs of average length', 'long legs',
                                'very long legs'],
    Measurement.ARMS_RELATION: ['very short arms', 'short arms', 'arms of average length', 'long arms',
                                'very long arms'],
    Measurement.SHOULDERS_RELATION: ['very narrow shoulders', 'narrow shoulders', 'average shoulder width',
                                     'broad shoulders', 'very broad shoulders'],
    Measurement.HIP_RELATION: ['very narrow hip', 'narrow hip', 'average hip width',
                               'broad hip', 'very broad hip'],
    Measurement.ARM_THICKNESS: ['very thin arms', 'thin arms', 'arms of average thickness', 'thick arms',
                                'very thick arms'],
    Measurement.LEG_THICKNESS: ['very thin legs', 'thin legs', 'legs of average thickness', 'thick legs',
                                'very thick legs'],
    Measurement.BUST_THICKNESS: ['very thin bust', 'thin bust', 'bust of average thickness', 'thick bust',
                                 'very thick bust'],
    Measurement.HIP_THICKNESS: ['very thin hip', 'thin hip', 'hip of average thickness', 'thick hip',
                                'very thick hip'],
    Measurement.WAIST_THICKNESS: ['very thin waist', 'thin waist', 'waist of average thickness', 'thick waist',
                                  'very thick waist']
}

"""
Avatar generation
"""


def generate_avatar_from_betas(input_betas):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smplx.create(MODEL_FOLDER, model_type=MODEL_TYPE, gender=GENDER).to(device)
    betas = torch.tensor([input_betas], dtype=torch.float32, device=device)
    expression = None

    output = model(betas=betas, expression=expression, return_verts=True)
    vertices = output.vertices.detach().squeeze()
    joints = output.joints.detach().squeeze()

    return model, output, vertices, joints


def generate_avatar_from_betas_batch(betas):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smplx.create(MODEL_FOLDER, model_type=MODEL_TYPE, gender=GENDER, ext='npz', num_betas=10,
                         batch_size=len(betas)).to(device)
    output = model(betas=betas)
    vertices = output.vertices

    return model, output, vertices


def generate_random_avatar(amplitude):
    betas = [0] * 10

    for i in range(0, 10):
        betas[i] = np.random.normal(0.0, 1.0) * .75 * amplitude

    return generate_avatar_from_betas(betas)


def generate_random_avatar_batch(num, amplitude):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smplx.create(MODEL_FOLDER, model_type=MODEL_TYPE, gender=GENDER, ext='npz', num_betas=10,
                         batch_size=num).to(device)
    batch_size = num
    num_betas = model.num_betas
    betas = torch.randn(batch_size, num_betas, dtype=torch.float32, device=device) * amplitude
    output = model(betas=betas)
    vertices = output.vertices

    return model, output, vertices, betas


def read_betas_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read().replace('\n', '')  # Lee el contenido y elimina los saltos de línea
            betas = list(
                map(float, content.split(',')))  # Convierte los valores a floats y los almacena en una lista
            return betas
    except FileNotFoundError:
        print(f"El archivo '{file_path}' no fue encontrado.")
        return None
    except Exception as e:
        print("Ocurrió un error al leer el archivo:", e)
        return None


"""
Evaluation
"""


def evaluate_measurement(value, ranges):
    range_index = 2

    if value <= ranges[0]:
        range_index = 0
    elif ranges[0] < value <= ranges[1]:
        range_index = 1
    elif ranges[1] < value <= ranges[2]:
        range_index = 2
    elif ranges[2] < value <= ranges[3]:
        range_index = 3
    elif value > ranges[3]:
        range_index = 4

    return range_index


def print_evaluation_debug(model, vertices, ranges):
    print("\n")
    print("-- MEASUREMENTS AND EVALUATION --")
    print("\n")
    for measurement_ in Measurement:
        if measurement_.value in ranges:
            value = get_measurement(model, vertices, measurement_)
            evaluation = evaluations_dictionary[measurement_][evaluate_measurement(value, ranges[measurement_.value])]
            print(str(measurement_.value) + ": " + str(value) + " - " + evaluation)


def get_evaluation_batch(model, vertices, ranges):
    evaluation_dict = {}  # Diccionario de tensores, cada tensor almacena N medidas de tipo measurement_
    measurement_values = get_measurements_batch(model, vertices).transpose(0, 1)
    i = 0
    test = measurement_values[i]
    for measurement_ in measurements_to_export:
        evaluation_dict[measurement_.value] = evaluate_measurements(measurement_values[i], ranges[measurement_.value])
        i += 1

    return evaluation_dict


def get_evaluation(measurements, ranges):
    evaluation_dict = {}  # Diccionario de tensores, cada tensor almacena N medidas de tipo measurement_

    for measurement_ in measurements:
        evaluation_dict[measurement_] = evaluate_measurements(measurements[measurement_], ranges[measurement_])

    return evaluation_dict


def get_evaluation_vector(betas, ranges):
    # Dadas unas betas, obtiene las medidas
    model, output, vertices = generate_avatar_from_betas_batch(betas)
    measurements = get_measurements_batch(model, vertices)

    evaluation_dict = {}  # Diccionario de tensores, cada tensor almacena N medidas de tipo measurement_

    for measurement_ in measurements:
        evaluation_dict[measurement_] = evaluate_measurements(measurements[measurement_], ranges[measurement_])

    evaluation_values = []
    num_values = len(evaluation_dict[next(iter(evaluation_dict))])
    for i in range(num_values):
        for key in evaluation_dict:
            evaluation_values.append(evaluation_dict[key][i].item())

    evaluation_tensor = torch.tensor(evaluation_values, device='cuda', dtype=torch.float32)
    evaluation_tensor_dim = evaluation_tensor.view(len(betas), len(ranges))
    return evaluation_tensor_dim


def get_evaluation_vector_expanded(betas, ranges):
    evaluation = get_evaluation_vector(betas, ranges)
    return vector_to_one_hot(evaluation, 5)


def vector_to_one_hot(matrix, num_classes):
    max_value = num_classes
    one_hot_matrix = torch.zeros((matrix.size(1), max_value), device="cuda")
    matrix = matrix.long()
    one_hot_matrix.scatter_(1, matrix, 1)
    return one_hot_matrix


def generate_description(evaluation):
    evaluation_batch = []
    for i in range(len(evaluation['height'])):
        evaluated_measurements = evaluation
        answer = "A"

        if evaluated_measurements[Measurement.HEIGHT.value][i] == 2:
            answer += "n"

        answer += " "
        answer += textual_expressions_english[Measurement.HEIGHT][evaluated_measurements[Measurement.HEIGHT.value][i]]

        answer += " person"

        first = True
        used_comma = False

        for measurement_ in measurements_to_export:
            if measurement_ is not Measurement.HEIGHT:
                if measurement_ is not Measurement.HEIGHT:
                    random_include = random.randint(1, 10)
                    if evaluated_measurements[measurement_.value][i] != 2 or random_include >= 9:
                        if first:
                            first = False
                            answer += " with "
                        else:
                            answer += ", "
                            used_comma = True
                        answer += textual_expressions_english[measurement_][
                            evaluated_measurements[measurement_.value][i]]

        if used_comma:
            answer = replace_last_comma(answer, "and")
        answer += "."
        evaluation_batch.append(answer)
    return evaluation_batch


def evaluate_measurements(measurements, ranges):
    # Crear un tensor de ceros para almacenar los resultados
    evaluation = torch.zeros_like(measurements, dtype=torch.int)

    # Iterar sobre los rangos
    for i in range(len(ranges) - 1):
        # Determinar los elementos que cumplen la condición para este rango
        mask = (measurements >= ranges[i]) & (measurements < ranges[i + 1])
        # Asignar el valor correspondiente en la evaluación
        evaluation[mask] = i + 1

    # Evaluar el último rango
    mask = measurements >= ranges[-1]
    evaluation[mask] = len(ranges)

    return evaluation

    vert_id_1 = vertex_distance_dictionary[measurement][0].value
    vert_id_2 = vertex_distance_dictionary[measurement][1].value
    distance = get_vert_distance(vertices, vert_id_1, vert_id_2)
    return distance


def replace_last_comma(original_string, word_to_use):
    last_comma_index = original_string.rfind(',')
    cadena = original_string
    # Si hay al menos una coma en la cadena
    if last_comma_index != -1:
        # Reemplaza la última coma con 'y'
        cadena = original_string[:last_comma_index] + ' ' + word_to_use + original_string[last_comma_index + 1:]

    return cadena


"""
Rendering
"""


def render_avatar(vertices, model, texture_path=None):
    vertices_list = vertices.detach().cpu().numpy().squeeze()
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.8, 0.2, 0.2, 0.7]
    mesh = trimesh.Trimesh(vertices_list, model.faces, vertex_colors=vertex_colors)

    if isinstance(mesh, trimesh.Trimesh):
        scene = trimesh.Scene(mesh)
    else:
        scene = mesh

    scene.show(
        viewer='gl',
        resolution=(600, 600),
        background=[0.7, 0.7, 0.7, 1.0],  
        window_title='Model Render' 
    )

"""
Get variables and measurements
"""


def get_vert_distance(vertices, vert_id_1, vert_id_2):
    vert_1 = vertices[:, vert_id_1, :]
    vert_2 = vertices[:, vert_id_2, :]
    dist_vector = vert_2 - vert_1
    dist_escalar = torch.norm(dist_vector, p=2, dim=-1)
    return dist_escalar


def get_measurement(model, vertices, measurement):
    measurement_type = measurement_types_dictionary[measurement]
    value = 0

    if measurement_type == MeasurementType.VERTEX_DISTANCE:
        value = get_measurement_distance(vertices, measurement)
    elif measurement_type == MeasurementType.RELATION:
        value = get_measurement_relation(model, vertices, measurement)
    elif measurement_type == MeasurementType.VOLUME:
        value = get_measurement_volume(model, vertices)
    elif measurement_type == MeasurementType.THICKNESS:
        value = get_measurement_thickness(vertices, measurement)

    return value


def get_measurement_distance(vertices, measurement):
    vert_id_1 = vertex_distance_dictionary[measurement][0].value
    vert_id_2 = vertex_distance_dictionary[measurement][1].value
    distance = get_vert_distance(vertices, vert_id_1, vert_id_2)
    return distance


def get_measurement_volume(model, vertices):
    faces = torch.tensor(model.faces.astype(np.int32), dtype=torch.int32, device='cuda')

    face_vertices = vertices[:, faces]

    edge1 = face_vertices[:, :, 1] - face_vertices[:, :, 0]
    edge2 = face_vertices[:, :, 2] - face_vertices[:, :, 0]

    normals = torch.cross(edge1, edge2, dim=2)

    volumes = torch.einsum('bfi,bfi->b', normals, face_vertices[:, :, 0]) / 6.0

    return torch.abs(volumes)


def get_measurement_relation(model, vertices, measurement):
    var_1 = measurement_relations_dictionary[measurement][0]
    var_2 = measurement_relations_dictionary[measurement][1]

    value_1 = get_measurement(model, vertices, var_1)
    value_2 = get_measurement(model, vertices, var_2)

    relation = value_1 / value_2
    return relation


def get_measurement_thickness(vertices, measurement):
    vertices_list = vertex_thickness_dictionary[measurement]
    position_1 = vertices[:, vertices_list[0]]
    position_2 = vertices[:, vertices_list[1]]
    position_3 = vertices[:, vertices_list[2]]
    position_4 = vertices[:, vertices_list[3]]

    distance_1 = position_2 - position_1
    distance_2 = position_4 - position_3

    norm_distance_1 = torch.norm(distance_1, dim=1, keepdim=True)
    norm_distance_2 = torch.norm(distance_2, dim=1, keepdim=True)

    thickness = torch.cat((norm_distance_1, norm_distance_2), dim=1)
    thickness = torch.sum(thickness, dim=1)

    return thickness


def print_all_measurements(model, vertices):
    print("\n")
    print("-- MEASUREMENTS --")
    print("\n")
    for measurement_ in Measurement:
        print(str(measurement_.value) + ": " + str(get_measurement(model, vertices, measurement_)))


def get_measurements_batch(model, vertices):
    measurements_list = []  # Lista de tensores, cada tensor almacena N medidas de tipo measurement_
    for measurement_ in measurements_to_export:
        measurements_list.append(get_measurement(model, vertices, measurement_))

    measurements_tensor = torch.stack(measurements_list).transpose(0, 1)
    return measurements_tensor


"""
Compute Error
"""


def compute_accuracy(expected_betas, generated_betas, ranges_data):
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
        generated_list = expected_betas
        expected_list = expected_betas
        expected_ranges = []
        min_value = min(generated_betas)
        max_value = max(generated_betas)
        # Se añaden el límite superior y límite inferior, que va desde el límite establecido hasta el último vañor o
        # crea un rango tan grande como el rango anterior
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

        return result_analysis