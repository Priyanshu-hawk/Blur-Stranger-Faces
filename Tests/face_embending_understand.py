# built-in dependencies
import time
from typing import Any, Dict, Optional, Union, List, Tuple

# 3rd party dependencies
import deepface.DeepFace
import numpy as np

# project dependencies
from deepface.modules import representation, detection
from deepface.commons.logger import Logger
import deepface
import cv2

logger = Logger()

model_name="ArcFace"
detector_backend="retinaface"
enforce_detection=True
align=True
expand_percentage=0
normalization="base"
anti_spoofing=False
silent=False



model = deepface.DeepFace.build_model(
        task="facial_recognition", model_name=model_name
    )

dims = model.output_shape

no_facial_area = {
    "x": None,
    "y": None,
    "w": None,
    "h": None,
    "left_eye": None,
    "right_eye": None,
}



def extract_embeddings_and_facial_areas(
        img_path: Union[str, np.ndarray, List[float]], index: int
    ) -> Tuple[List[List[float]], List[dict]]:
        """
        Extracts facial embeddings and corresponding facial areas from an
        image or returns pre-calculated embeddings.

        Depending on the type of img_path, the function either extracts
        facial embeddings from the provided image
        (via a path or NumPy array) or verifies that the input is a list of
        pre-calculated embeddings and validates them.

        Args:
            img_path (Union[str, np.ndarray, List[float]]):
                - A string representing the file path to an image,
                - A NumPy array containing the image data,
                - Or a list of pre-calculated embedding values (of type `float`).
            index (int): An index value used in error messages and logging
            to identify the number of the image.

        Returns:
            Tuple[List[List[float]], List[dict]]:
                - A list containing lists of facial embeddings for each detected face.
                - A list of dictionaries where each dictionary contains facial area information.
        """
        if isinstance(img_path, list):
            # given image is already pre-calculated embedding
            if not all(isinstance(dim, float) for dim in img_path):
                raise ValueError(
                    f"When passing img{index}_path as a list,"
                    " ensure that all its items are of type float."
                )

            if silent is False:
                logger.warn(
                    f"You passed {index}-th image as pre-calculated embeddings."
                    "Please ensure that embeddings have been calculated"
                    f" for the {model_name} model."
                )

            if len(img_path) != dims:
                raise ValueError(
                    f"embeddings of {model_name} should have {dims} dimensions,"
                    f" but {index}-th image has {len(img_path)} dimensions input"
                )

            img_embeddings = [img_path]
            img_facial_areas = [no_facial_area]
        else:
            try:
                img_embeddings, img_facial_areas = __extract_faces_and_embeddings(
                    img_path=img_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=enforce_detection,
                    align=align,
                    expand_percentage=expand_percentage,
                    normalization=normalization,
                    anti_spoofing=anti_spoofing,
                )
            except ValueError as err:
                raise ValueError(f"Exception while processing img{index}_path") from err
        return img_embeddings, img_facial_areas


def __extract_faces_and_embeddings(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
) -> Tuple[List[List[float]], List[dict]]:
    """
    Extract facial areas and find corresponding embeddings for given image
    Returns:
        embeddings (List[float])
        facial areas (List[dict])
    """
    embeddings = []
    facial_areas = []

    img_objs = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )

    # find embeddings for each face
    for img_obj in img_objs:
        if anti_spoofing is True and img_obj.get("is_real", True) is False:
            raise ValueError("Spoof detected in given image.")
        img_embedding_obj = representation.represent(
            img_path=img_obj["face"],
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )
        # already extracted face given, safe to access its 1st item
        img_embedding = img_embedding_obj[0]["embedding"]
        embeddings.append(img_embedding)
        facial_areas.append(img_obj["facial_area"])

    return embeddings, facial_areas


def find_cosine_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find cosine distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    a = np.dot(source_representation, test_representation)
    b = np.linalg.norm(source_representation)
    c = np.linalg.norm(test_representation)
    return 1 - a / (b * c)


def find_euclidean_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find euclidean distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated euclidean distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    return np.linalg.norm(source_representation - test_representation)


def l2_normalize(x: Union[np.ndarray, list]) -> np.ndarray:
    """
    Normalize input vector with l2
    Args:
        x (np.ndarray or list): given vector
    Returns:
        y (np.ndarray): l2 normalized vector
    """
    if isinstance(x, list):
        x = np.array(x)
    norm = np.linalg.norm(x)
    return x if norm == 0 else x / norm

def find_distance(
    alpha_embedding: Union[np.ndarray, list],
    beta_embedding: Union[np.ndarray, list],
    distance_metric: str,
) -> np.float64:
    """
    Wrapper to find distance between vectors according to the given distance metric
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if distance_metric == "cosine":
        distance = find_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_l2":
        distance = find_euclidean_distance(
            l2_normalize(alpha_embedding), l2_normalize(beta_embedding)
        )
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    return distance

def find_threshold(model_name: str, distance_metric: str) -> float:
    """
    Retrieve pre-tuned threshold values for a model and distance metric pair
    Args:
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        distance_metric (str): distance metric name. Options are cosine, euclidean
            and euclidean_l2.
    Returns:
        threshold (float): threshold value for that model name and distance metric
            pair. Distances less than this threshold will be classified same person.
    """

    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

    thresholds = {
        # "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}, # 2622d
        "VGG-Face": {
            "cosine": 0.68,
            "euclidean": 1.17,
            "euclidean_l2": 1.17,
        },  # 4096d - tuned with LFW
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
        "GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "euclidean_l2": 1.10},
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

    return threshold


if __name__ == "__main__":
    # img1 = "../test2.jpeg"
    img1 = "../IMG_9798.jpeg"
    img2 = "../IMG_9830.jpeg"
    

    distance_metric = "cosine"
    threshold = 0.40 ## custom threshold for cosine distance

    # print("Finding distance between two images")
    img2 = cv2.imread(img2)
    # img2_show = img2_show.resize(1280, 720)

    img1_embd, face1_area = extract_embeddings_and_facial_areas(img1, 1)
    img2_embd, face2_area = extract_embeddings_and_facial_areas(img2, 2)


    # print(img2_embd)
    # print(face2_area)

    print("Embeddings found")

    for ig2_emb, f2_area in zip(img2_embd, face2_area):
        ig2_emb = [ig2_emb]
        f2_area = [f2_area]

        min_distance, min_idx, min_idy = float("inf"), None, None
        for idx, img1_embedding in enumerate(img1_embd):
            for idy, img2_embedding in enumerate(ig2_emb):
                distance = find_distance(img1_embedding, img2_embedding, distance_metric="cosine")

                # print("Distance: ", distance)
                # print("Min Distance: ", min_distance)

                if distance < min_distance:
                    min_distance, min_idx, min_idy = distance, idx, idy
        
        threshold = threshold or find_threshold(model_name, distance_metric)
        distance = float(min_distance)

        print("Distance: ", distance)
        print("Threshold: ", threshold)

        facial_areas = (
            no_facial_area if min_idx is None else face1_area[min_idx],
            no_facial_area if min_idy is None else f2_area[min_idy],
        )
        
        cv2.rectangle(img2, (facial_areas[1]['x'], facial_areas[1]['y']), (facial_areas[1]['x']+facial_areas[1]['w'], facial_areas[1]['y']+facial_areas[1]['h']), (0, 255, 0), 2)
        cv2.putText(img2, "same: {}".format(distance <= threshold), (facial_areas[1]['x'], facial_areas[1]['y'] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (36,255,12), 2)
        import time
        cv2.imwrite('output{}.jpeg'.format(time.time()), img2)
        print("Faces are same: ", distance <= threshold)
        print("-"*20)


    # img = np.array(cv2.imread(img1))
    # img = cv2.imread(img2)
    # for f_fet in face2_area:
    #     tmp_img = img[f_fet['y']:f_fet['y']+f_fet['h'], f_fet['x']:f_fet['x']+f_fet['w']]
    #     cv2.imshow('img', tmp_img)
    #     cv2.waitKey(0)
    
    # img_ext = __extract_faces_and_embeddings(img2, model_name="ArcFace", detector_backend="retinaface")
    # print(img_ext)