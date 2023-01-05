import tensorflow as tf
import numpy as np
import cv2
import pyrealsense2 as rs
import timeit
import math
from vpython import *


class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

        # align depth to color
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # intrinsic을 위한
        depth_frame_for_intrinsic = depth_frame

        if not depth_frame or not color_frame:
            # if there is no frame, probably camera not connected
            print("Error, impossibile to get the frame, make sure that the camera is correctly connected")
            return False, None, None

        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)

        # create colormap to show the depth of the Objects
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())

        # Convert images to numpy arrays
        # distance = depth_frame.get_distance(int(50),int(50))
        # print("distance",distance)
        depth_image = np.asanyarray(filled_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # if depth and color resolution are different, resize color images to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        return True, color_image, depth_image, depth_colormap, depth_frame_for_intrinsic

    def release(self):
        self.pipeline.stop()


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape  # y coord, x coord, channel coordinates
    shaped = np.squeeze(
        np.multiply(keypoints, [y, x, 1]))  # multiply our frame shape를 해줘서 실제 좌표값 confidence는 transform 해주기 싫으니 1 곱하구

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def get_3d_coordinates(keypoints):
    keypoints_position = keypoints_with_scores_color[0][0][keypoints]
    keypoints_position_int = np.array(keypoints_position[:2] * [720, 1280]).astype(int).tolist()
    keypoints_position_int.reverse()  # movenet 결과는 [y,x]로 반환하기 때문에

    distance = depth_frame_for_intrinsic.get_distance(keypoints_position_int[0], keypoints_position_int[1])

    keypoints_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, keypoints_position_int, distance)

    keypoints_3d[1], keypoints_3d[2] = keypoints_3d[2], -keypoints_3d[1]

    return keypoints_3d
    # specified in meters, with the coordinate [0,0,0] referring to the center of the physical imager(Left IR)

def get_own_xyz(vector1, vector2):
    z = np.array([vector1[0]-vector2[0], vector1[1]-vector2[1], vector1[2]-vector2[2]])
    y = np.array([vector1[1]-vector2[1], vector2[0]-vector1[0], 0])
    x = np.cross(z,y)
    return x, y, z

def make_unit_vector(vector):
    vector = vector/np.linalg.norm(vector)
    return vector

def point_distance(point1, point2):
    distance = math.sqrt(pow(point2[0]-point1[0],2)+pow(point2[1]-point1[1],2)+pow(point2[2]-point1[2],2))
    return distance


# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="C:\\Users\\MinLab\Desktop\\internship_JS\\lite-model_movenet_singlepose_thunder_3.tflite")
interpreter.allocate_tensors()

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

dc = DepthCamera()
# 여기는 while문 밖에 위치해야할듯 임의로 처음 simul하는대로
human_head_simul = box(pos=vector(0, 0, 0), axis=vector(0, 0, 1),
                       size=vector(0, 0, 0))
human_right_arm1_simul = box(pos=vector(0, 0, 0), axis=vector(0, 0, 0),
                             size=vector(0, 0, 0))
human_right_arm2_simul = box(pos=vector(0, 0, 0), axis=vector(0, 0, 0),
                             size=vector(0, 0, 0))
human_left_arm1_simul = box(pos=vector(0, 0, 0), axis=vector(0, 0, 0),
                            size=vector(0, 0, 0))
human_left_arm2_simul = box(pos=vector(0, 0, 0), axis=vector(0, 0, 0),
                            size=vector(0, 0, 0))
human_body_simul = box(pos=vector(0, 0, 0), axis=vector(0, 0, 0),
                       size=vector(0, 0, 0))
while True:
    start_t = timeit.default_timer()  # start time save

    ret, color_frame, depth_frame, depth_colormap, depth_frame_for_intrinsic = dc.get_frame()

    # Reshape image
    img_color = color_frame.copy()
    img_color = tf.image.resize_with_pad(np.expand_dims(img_color, axis=0), 256, 256)
    input_image_color = tf.cast(img_color, dtype=tf.float32)

    # Setup input and output
    input_details_color = interpreter.get_input_details()
    output_details_color = interpreter.get_output_details()

    # Make predictions
    interpreter.set_tensor(input_details_color[0]['index'], np.array(input_image_color))
    interpreter.invoke()
    keypoints_with_scores_color = interpreter.get_tensor(output_details_color[0]['index'])

    # Rendering
    draw_connections(color_frame, keypoints_with_scores_color, EDGES, 0.4)
    draw_keypoints(color_frame, keypoints_with_scores_color, 0.4)

    # Pixel to 3d coordinates from depth images
    depth_intrin = depth_frame_for_intrinsic.profile.as_video_stream_profile().intrinsics

    # get distance from each keypoints
    # 17 keypoints (in the order of: [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder,
    #                                left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee,
    #                                right knee, left ankle, right ankle]  index 0부터 시작
    nose = get_3d_coordinates(0)
    left_shoulder = get_3d_coordinates(5)
    right_shoulder = get_3d_coordinates(6)
    left_elbow = get_3d_coordinates(7)
    right_elbow = get_3d_coordinates(8)
    left_hip = get_3d_coordinates(11)
    right_hip = get_3d_coordinates(12)
    left_wrist = get_3d_coordinates(9)
    right_wrist = get_3d_coordinates(10)

    human_head_data = [nose, [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.15, 0.15, 0.24]]
    human_right_arm1_data = [[(right_shoulder[0] + right_elbow[0]) / 2, (right_shoulder[1] + right_elbow[1]) / 2,
                              (right_shoulder[2] + right_elbow[2]) / 2],
                             make_unit_vector(get_own_xyz(right_shoulder, right_elbow))[0].tolist(),
                             make_unit_vector(get_own_xyz(right_shoulder, right_elbow))[1].tolist(),
                             make_unit_vector(get_own_xyz(right_shoulder, right_elbow))[2].tolist(),
                             [0.05, 0.05, point_distance(right_shoulder, right_elbow)]]
    human_right_arm2_data = [[(right_elbow[0] + right_wrist[0]) / 2, (right_elbow[1] + right_wrist[1]) / 2,
                              (right_elbow[2] + right_wrist[2]) / 2],
                             make_unit_vector(get_own_xyz(right_elbow, right_wrist))[0].tolist(),
                             make_unit_vector(get_own_xyz(right_elbow, right_wrist))[1].tolist(),
                             make_unit_vector(get_own_xyz(right_elbow, right_wrist))[2].tolist(),
                             [0.05, 0.05, point_distance(right_elbow, right_wrist)]]
    human_left_arm1_data = [[(left_shoulder[0] + left_elbow[0]) / 2, (left_shoulder[1] + left_elbow[1]) / 2,
                             (left_shoulder[2] + left_elbow[2]) / 2],
                            make_unit_vector(get_own_xyz(left_shoulder, left_elbow))[0].tolist(),
                            make_unit_vector(get_own_xyz(left_shoulder, left_elbow))[1].tolist(),
                            make_unit_vector(get_own_xyz(left_shoulder, left_elbow))[2].tolist(),
                            [0.05, 0.05, point_distance(left_shoulder, left_elbow)]]
    human_left_arm2_data = [
        [(left_elbow[0] + left_wrist[0]) / 2, (left_elbow[1] + left_wrist[1]) / 2, (left_elbow[2] + left_wrist[2]) / 2],
        make_unit_vector(get_own_xyz(left_elbow, left_wrist))[0].tolist(),
        make_unit_vector(get_own_xyz(left_elbow, left_wrist))[1].tolist(),
        make_unit_vector(get_own_xyz(left_elbow, left_wrist))[2].tolist(),
        [0.05, 0.05, point_distance(left_elbow, left_wrist)]]
    human_body_data = [[(right_shoulder[0] + left_shoulder[0]) / 2, (right_shoulder[1] + left_shoulder[1]) / 2,
                        (right_shoulder[2] + right_hip[2]) / 2], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                       [point_distance(right_shoulder, left_shoulder), point_distance(right_shoulder, left_shoulder),
                        point_distance(right_shoulder, right_hip)]]

    human_head = np.array(human_head_data)
    human_right_arm1 = np.array(human_right_arm1_data)
    human_right_arm2 = np.array(human_right_arm2_data)
    human_left_arm1 = np.array(human_left_arm1_data)
    human_left_arm2 = np.array(human_left_arm2_data)
    human_body = np.array(human_body_data)

    human_head_simul.pos = vector(nose[0], nose[1], nose[2])
    human_head_simul.axis = vector(0, 0, 1)
    human_head_simul.size = vector(0.24, 0.15, 0.15)

    human_right_arm1_simul.pos = vector((right_shoulder[0] + right_elbow[0]) / 2,
                                        (right_shoulder[1] + right_elbow[1]) / 2,
                                        (right_shoulder[2] + right_elbow[2]) / 2)
    human_right_arm1_simul.axis = vector(make_unit_vector(get_own_xyz(right_shoulder, right_elbow))[2][0],
                                         make_unit_vector(get_own_xyz(right_shoulder, right_elbow))[2][1],
                                         make_unit_vector(get_own_xyz(right_shoulder, right_elbow))[2][2])
    human_right_arm1_simul.size = vector(point_distance(right_shoulder, right_elbow), 0.05, 0.05)

    human_right_arm2_simul.pos = vector((right_elbow[0] + right_wrist[0]) / 2, (right_elbow[1] + right_wrist[1]) / 2,
                                        (right_elbow[2] + right_wrist[2]) / 2)
    human_right_arm2_simul.axis = vector(make_unit_vector(get_own_xyz(right_elbow, right_wrist))[2][0],
                                         make_unit_vector(get_own_xyz(right_elbow, right_wrist))[2][1],
                                         make_unit_vector(get_own_xyz(right_elbow, right_wrist))[2][2])
    human_right_arm2_simul.size = vector(point_distance(right_elbow, right_wrist), 0.05, 0.05)

    human_left_arm1_simul.pos = vector((left_shoulder[0] + left_elbow[0]) / 2, (left_shoulder[1] + left_elbow[1]) / 2,
                                       (left_shoulder[2] + left_elbow[2]) / 2)
    human_left_arm1_simul.axis = vector(make_unit_vector(get_own_xyz(left_shoulder, left_elbow))[2][0],
                                        make_unit_vector(get_own_xyz(left_shoulder, left_elbow))[2][1],
                                        make_unit_vector(get_own_xyz(left_shoulder, left_elbow))[2][2])
    human_left_arm1_simul.size = vector(point_distance(left_shoulder, left_elbow), 0.05, 0.05)

    human_left_arm2_simul.pos = vector((left_elbow[0] + left_wrist[0]) / 2, (left_elbow[1] + left_wrist[1]) / 2,
                                       (left_elbow[2] + left_wrist[2]) / 2)
    human_left_arm2_simul.axis = vector(make_unit_vector(get_own_xyz(left_elbow, left_wrist))[2][0],
                                        make_unit_vector(get_own_xyz(left_elbow, left_wrist))[2][1],
                                        make_unit_vector(get_own_xyz(left_elbow, left_wrist))[2][2])
    human_left_arm2_simul.size = vector(point_distance(left_elbow, left_wrist), 0.05, 0.05)

    human_body_simul.pos = vector((right_shoulder[0] + left_shoulder[0]) / 2,
                                  (right_shoulder[1] + left_shoulder[1]) / 2, (right_shoulder[2] + right_hip[2]) / 2)
    human_body_simul.axis = vector(0, 0, 1)
    human_body_simul.size = vector(point_distance(right_shoulder, right_hip),
                                   point_distance(right_shoulder, left_shoulder),
                                   point_distance(right_shoulder, left_shoulder))

    cv2.imshow('Depth frame', depth_frame)
    cv2.imshow('Color frame', color_frame)
    cv2.imshow('depth_colormap', depth_colormap)

    terminate_t = timeit.default_timer()  # finish time save
    FPS = int(1. / (terminate_t - start_t))
    #     print(FPS)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
dc.release()
cv2.destroyAllwindows()
