import sys
import os
from scipy.spatial.transform import Rotation as R
import os
import cv2
import rospy
import rosbag
import progressbar
from tf2_msgs.msg import TFMessage
from datetime import datetime
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Imu, PointField, NavSatFix
import sensor_msgs.point_cloud2 as pcl2
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform
from cv_bridge import CvBridge
import numpy as np
import argparse
from pathlib import Path

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
import av2.utils.io as av2_io
import av2.geometry.geometry as geometry_utils
import av2.utils.dense_grid_interpolation as dense_grid_interpolation


def save_dynamic_tf(bag, av2_dataloader, lidar_fpaths, log_id):
    print("Exporting time dependent transformations")
    for lidar_fpath in lidar_fpaths:
        timestamp_ns = float(lidar_fpath.stem)
        dt = datetime.fromtimestamp(timestamp_ns / 1000000000)
        timestamp = rospy.Time.from_sec(float(dt.strftime("%s.%f")))
        map2lidar = (
            av2_dataloader.get_city_SE3_ego(
                log_id=log_id, timestamp_ns=int(lidar_fpath.stem)
            )
            .inverse()
            .transform_matrix
        )
        tf_msg = TFMessage()
        tf_stamped = TransformStamped()
        tf_stamped.header.stamp = timestamp
        tf_stamped.header.frame_id = "map"
        tf_stamped.child_frame_id = "base_link"

        t = map2lidar[0:3, 3]
        q = R.from_matrix(map2lidar[:3, :3]).as_quat()
        transform = Transform()

        transform.translation.x = t[0]
        transform.translation.y = t[1]
        transform.translation.z = t[2]

        transform.rotation.x = q[0]
        transform.rotation.y = q[1]
        transform.rotation.z = q[2]
        transform.rotation.w = q[3]

        tf_stamped.transform = transform
        tf_msg.transforms.append(tf_stamped)

        bag.write("/tf", tf_msg, tf_msg.transforms[0].header.stamp)


def save_camera_data(bag, av2_cam, cam_fpaths, camera, camera_frame_id, topic, args):
    print("Exporting camera {}".format(camera))
    calib = CameraInfo()
    calib.header.frame_id = camera_frame_id
    if args.full:
        calib.width = av2_cam.width_px
        calib.height = av2_cam.height_px
    else:
        calib.width = 640
        calib.height = 480
    calib.distortion_model = "plumb_bob"
    cam_K = av2_cam.intrinsics.K
    if not args.full:
        cam_K[0, 0] = cam_K[0, 0] / 3.2  # fx
        cam_K[1, 1] = cam_K[1, 1] / 3.2  # fy
        cam_K[0, 2] = cam_K[0, 2] / 3.2  # cx
        cam_K[1, 2] = cam_K[1, 2] / 3.2  # cy
    calib.K = cam_K.reshape(9).tolist()
    # TODO: calib.D calib.P
    bridge = CvBridge()

    bar = progressbar.ProgressBar()
    for cam_fpath in bar(cam_fpaths):
        cv_img = av2_io.read_img(cam_fpath)
        if args.full:
            encoding = "bgr8"
        else:
            cv_img = cv2.resize(cv_img, (640, 480))
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            encoding = "mono8"
        image_message = bridge.cv2_to_imgmsg(cv_img, encoding=encoding)
        image_message.header.frame_id = camera_frame_id
        timestamp_ns = float(cam_fpath.stem)
        dt = datetime.fromtimestamp(timestamp_ns / 1000000000)
        image_message.header.stamp = rospy.Time.from_sec(
            float(datetime.strftime(dt, "%s.%f"))
        )
        calib.header.stamp = image_message.header.stamp
        bag.write(topic + "/image_raw", image_message, t=image_message.header.stamp)
        bag.write(topic + "/camera_info", calib, t=calib.header.stamp)


def save_lidar_data(bag, lidar_fpaths, lidar_frame_id, lidar_topic):
    print("Exporting lidar data")
    bar = progressbar.ProgressBar()
    for lidar_fpath in lidar_fpaths:
        # create header
        header = Header()
        header.frame_id = lidar_frame_id

        timestamp_ns = float(lidar_fpath.stem)
        dt = datetime.fromtimestamp(timestamp_ns / 1000000000)
        header.stamp = rospy.Time.from_sec(float(datetime.strftime(dt, "%s.%f")))

        # fill pcl msg
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        np_pc = av2_io.read_lidar_sweep(lidar_fpath)
        pcl_msg = pcl2.create_cloud(header, fields, np.asarray(np_pc))
        bag.write(lidar_topic + "/pointcloud", pcl_msg, t=pcl_msg.header.stamp)


def get_static_transform(from_frame_id, to_frame_id, transform):
    t = transform[0:3, 3]
    q = R.from_matrix(transform[:3, :3]).as_quat()
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = from_frame_id
    tf_msg.child_frame_id = to_frame_id
    tf_msg.transform.translation.x = float(t[0])
    tf_msg.transform.translation.y = float(t[1])
    tf_msg.transform.translation.z = float(t[2])
    tf_msg.transform.rotation.x = float(q[0])
    tf_msg.transform.rotation.y = float(q[1])
    tf_msg.transform.rotation.z = float(q[2])
    tf_msg.transform.rotation.w = float(q[3])
    return tf_msg


def inv(transform):
    "Invert rigid body transformation matrix"
    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    t_inv = -1 * R.T.dot(t)
    transform_inv = np.eye(4)
    transform_inv[0:3, 0:3] = R.T
    transform_inv[0:3, 3] = t_inv
    return transform_inv


def save_static_transforms(bag, transforms, lidar_fpaths):
    print("Exporting static transformations")
    tfm = TFMessage()
    for transform in transforms:
        t = get_static_transform(
            from_frame_id=transform[0], to_frame_id=transform[1], transform=transform[2]
        )
        tfm.transforms.append(t)
    for lidar_fpath in lidar_fpaths:
        timestamp_ns = float(lidar_fpath.stem)
        dt = datetime.fromtimestamp(timestamp_ns / 1000000000)
        time = rospy.Time.from_sec(float(dt.strftime("%s.%f")))
        for i in range(len(tfm.transforms)):
            tfm.transforms[i].header.stamp = time
        bag.write("/tf_static", tfm, t=time)


def run_argoverse2bag():
    parser = argparse.ArgumentParser(
        description="Convert argoverse2 dataset to ROS bag file the easy way!"
    )
    parser.add_argument(
        "data_root",
        nargs="?",
        default=os.getcwd(),
        help="base directory of the dataset, if no directory passed the deafult is current working directory",
    )
    parser.add_argument("--log_id", help="log id of the dataset")
    parser.add_argument("--full", action="store_true", help="save full data")
    args = parser.parse_args()

    bridge = CvBridge()
    compression = rosbag.Compression.NONE
    # compression = rosbag.Compression.BZ2
    # compression = rosbag.Compression.LZ4

    cameras = [
        (0, "ring_front_center", "/argoverse2/ring_front_center"),
        (1, "ring_front_right", "/argoverse2/ring_front_right"),
        (2, "ring_rear_right", "/argoverse2/ring_rear_right"),
        (3, "ring_side_right", "/argoverse2/ring_side_right"),
        (4, "ring_front_left", "/argoverse2/ring_front_left"),
        (5, "ring_rear_left", "/argoverse2/ring_rear_left"),
        (6, "ring_side_left", "/argoverse2/ring_side_left"),
    ]

    lidar_frame_id = "lidar_link"
    lidar_topic = "/argoverse2/lidar"

    if args.full:
        bag = rosbag.Bag(
            "argoverse2_{}_full.bag".format(args.log_id), "w", compression=compression
        )
    else:
        bag = rosbag.Bag(
            "argoverse2_{}.bag".format(args.log_id), "w", compression=compression
        )

    av2_dataloader = AV2SensorDataLoader(
        data_dir=Path(args.data_root), labels_dir=Path(args.data_root)
    )
    av2_cams = [
        av2_dataloader.get_log_pinhole_camera(args.log_id, cameras[i][1])
        for i in range(7)
    ]
    lidar_extrinsics = np.eye(4)

    # tf_static
    transforms = [
        ("base_link", lidar_frame_id, lidar_extrinsics),
        ("base_link", cameras[0][1], av2_cams[0].extrinsics),
        ("base_link", cameras[1][1], av2_cams[1].extrinsics),
        ("base_link", cameras[2][1], av2_cams[2].extrinsics),
        ("base_link", cameras[3][1], av2_cams[3].extrinsics),
        ("base_link", cameras[4][1], av2_cams[4].extrinsics),
        ("base_link", cameras[5][1], av2_cams[5].extrinsics),
        ("base_link", cameras[6][1], av2_cams[6].extrinsics),
    ]

    try:
        # Export
        lidar_fpaths = av2_dataloader.get_ordered_log_lidar_fpaths(args.log_id)
        save_static_transforms(bag, transforms, lidar_fpaths)
        save_dynamic_tf(bag, av2_dataloader, lidar_fpaths, log_id=args.log_id)
        for camera in cameras:
            cam_fpaths = av2_dataloader.get_ordered_log_cam_fpaths(
                args.log_id, camera[1]
            )
            save_camera_data(
                bag,
                av2_cams[camera[0]],
                cam_fpaths,
                camera=camera[0],
                camera_frame_id=camera[1],
                topic=camera[2],
                args=args,
            )
            # save only front center camera
            if not args.full:
                break
        if args.full:
            save_lidar_data(bag, lidar_fpaths, lidar_frame_id, lidar_topic)
    finally:
        print("## OVERVIEW ##")
        print(bag)
        bag.close()


def main():
    run_argoverse2bag()


if __name__ == "__main__":
    main()
